Attribute VB_Name = "MarketLens"
'==============================================================================
' MarketLens.bas  -  VBA module for MarketLens.xlsm
'
' 1) Worksheet functions (use them in any cell like a normal Excel formula):
'      =BS_PRICE(S, K, T, r, sigma, [type], [q])
'      =BS_GREEK("delta", S, K, T, r, sigma, [type], [q])
'           greek = "delta" | "gamma" | "vega" | "theta" | "rho" | "price"
'      =BS_IMPLIED_VOL(price, S, K, T, r, [type], [q])
'    type = "call" (default) or "put";  q = dividend yield (default 0)
'    Units: vega per 1 vol point, theta per calendar day, rho per 1% rate.
'
' 2) Macros:
'      RefreshVaR        - reads the yellow inputs on sheet "VaR Macro" and the
'                          daily returns on sheet "Returns", recomputes
'                          historical and parametric VaR / ES, writes the green
'                          results table and rebuilds two charts.
'      AddRefreshButton  - one-off helper: puts a "Refresh VaR" button on the
'                          "VaR Macro" sheet that runs RefreshVaR.
'
' The Python package (marketlens/risk.py, marketlens/options.py) uses the
' same formulas, so the numbers can be checked against each other.
'==============================================================================
Option Explicit

Private Const PI As Double = 3.14159265358979

'------------------------------------------------------------------------------
' Normal distribution helpers
'------------------------------------------------------------------------------
Private Function NormCdf(ByVal x As Double) As Double
    ' Standard normal CDF N(x) = P(Z <= x), from Excel's built-in NORM.S.DIST
    NormCdf = Application.WorksheetFunction.Norm_S_Dist(x, True)
End Function

Private Function NormPdf(ByVal x As Double) As Double
    ' Standard normal density phi(x) = exp(-x^2/2) / sqrt(2*pi)
    NormPdf = Exp(-0.5 * x * x) / Sqr(2 * PI)
End Function

Private Function IsCall(ByVal optType As String) As Boolean
    Select Case LCase$(Trim$(optType))
        Case "call", "c": IsCall = True
        Case "put", "p": IsCall = False
        Case Else: Err.Raise vbObjectError + 1, , "type must be ""call"" or ""put"""
    End Select
End Function

'------------------------------------------------------------------------------
' Black-Scholes price
'   d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),  d2 = d1 - sigma sqrt T
'   Call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
'   Put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
'------------------------------------------------------------------------------
Public Function BS_PRICE(ByVal S As Double, ByVal K As Double, ByVal T As Double, _
                         ByVal r As Double, ByVal sigma As Double, _
                         Optional ByVal optType As String = "call", _
                         Optional ByVal q As Double = 0) As Variant
    On Error GoTo Fail
    If S <= 0 Or K <= 0 Or T <= 0 Or sigma <= 0 Then GoTo Fail

    Dim d1 As Double, d2 As Double
    d1 = (Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Sqr(T))
    d2 = d1 - sigma * Sqr(T)

    If IsCall(optType) Then
        BS_PRICE = S * Exp(-q * T) * NormCdf(d1) - K * Exp(-r * T) * NormCdf(d2)
    Else
        BS_PRICE = K * Exp(-r * T) * NormCdf(-d2) - S * Exp(-q * T) * NormCdf(-d1)
    End If
    Exit Function
Fail:
    BS_PRICE = CVErr(xlErrValue)      ' shows #VALUE! for bad inputs
End Function

'------------------------------------------------------------------------------
' Black-Scholes Greeks
'------------------------------------------------------------------------------
Public Function BS_GREEK(ByVal greek As String, ByVal S As Double, ByVal K As Double, _
                         ByVal T As Double, ByVal r As Double, ByVal sigma As Double, _
                         Optional ByVal optType As String = "call", _
                         Optional ByVal q As Double = 0) As Variant
    On Error GoTo Fail
    If S <= 0 Or K <= 0 Or T <= 0 Or sigma <= 0 Then GoTo Fail

    Dim d1 As Double, d2 As Double, sqT As Double, eqT As Double, erT As Double
    Dim callFlag As Boolean
    sqT = Sqr(T)
    eqT = Exp(-q * T)
    erT = Exp(-r * T)
    d1 = (Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    callFlag = IsCall(optType)

    Select Case LCase$(Trim$(greek))
        Case "price"
            BS_GREEK = BS_PRICE(S, K, T, r, sigma, optType, q)
        Case "delta"     ' dV/dS
            If callFlag Then
                BS_GREEK = eqT * NormCdf(d1)
            Else
                BS_GREEK = -eqT * NormCdf(-d1)
            End If
        Case "gamma"     ' d2V/dS2 (same for call and put)
            BS_GREEK = eqT * NormPdf(d1) / (S * sigma * sqT)
        Case "vega"      ' dV/dsigma per 1 vol point (same for call and put)
            BS_GREEK = S * eqT * NormPdf(d1) * sqT / 100
        Case "theta"     ' dV/dt per calendar day
            Dim decay As Double
            decay = -S * eqT * NormPdf(d1) * sigma / (2 * sqT)
            If callFlag Then
                BS_GREEK = (decay - r * K * erT * NormCdf(d2) + q * S * eqT * NormCdf(d1)) / 365
            Else
                BS_GREEK = (decay + r * K * erT * NormCdf(-d2) - q * S * eqT * NormCdf(-d1)) / 365
            End If
        Case "rho"       ' dV/dr per 1% move in rates
            If callFlag Then
                BS_GREEK = K * T * erT * NormCdf(d2) / 100
            Else
                BS_GREEK = -K * T * erT * NormCdf(-d2) / 100
            End If
        Case Else
            GoTo Fail
    End Select
    Exit Function
Fail:
    BS_GREEK = CVErr(xlErrValue)
End Function

'------------------------------------------------------------------------------
' Implied volatility: Newton-Raphson with a bisection fallback
'   Newton step:  sigma <- sigma - (BS(sigma) - price) / vega
'   Start at the Manaster-Koehler point sqrt(2|ln(F/K)|/T) (or the ATM guess),
'   where Newton converges for any strike. If vega is ~0 or a step leaves
'   (0.0001, 5), fall back to bisection: price rises with sigma, so halving
'   an interval that brackets the price always works.
'   Returns #NUM! if the price breaks no-arbitrage bounds or has no time value.
'------------------------------------------------------------------------------
Public Function BS_IMPLIED_VOL(ByVal price As Double, ByVal S As Double, ByVal K As Double, _
                               ByVal T As Double, ByVal r As Double, _
                               Optional ByVal optType As String = "call", _
                               Optional ByVal q As Double = 0) As Variant
    On Error GoTo Fail
    Const TOL As Double = 0.0000000001
    Const LO As Double = 0.0001
    Const HI As Double = 5#

    Dim callFlag As Boolean, lower As Double, upper As Double
    Dim dS As Double, dK As Double
    callFlag = IsCall(optType)
    dS = S * Exp(-q * T)
    dK = K * Exp(-r * T)
    If callFlag Then
        lower = Application.WorksheetFunction.Max(dS - dK, 0): upper = dS
    Else
        lower = Application.WorksheetFunction.Max(dK - dS, 0): upper = dK
    End If
    If price <= lower Or price >= upper Then GoTo NoSolution
    If price - lower < 0.000001 * S Then GoTo NoSolution    ' no time value

    Dim sigma As Double, diff As Double, vega As Double, stepTo As Double, i As Long
    Dim F As Double
    F = S * Exp((r - q) * T)
    sigma = Application.WorksheetFunction.Max(Sqr(2 * PI / T) * price / S, _
                                              Sqr(2 * Abs(Log(F / K)) / T))
    If sigma < 0.05 Then sigma = 0.05
    If sigma > 3 Then sigma = 3

    For i = 1 To 100
        diff = BS_PRICE(S, K, T, r, sigma, optType, q) - price
        If Abs(diff) < TOL Then
            BS_IMPLIED_VOL = sigma
            Exit Function
        End If
        vega = BS_GREEK("vega", S, K, T, r, sigma, optType, q) * 100
        If vega < 0.00000001 Then Exit For
        stepTo = sigma - diff / vega
        If stepTo <= LO Or stepTo >= HI Then Exit For
        sigma = stepTo
    Next i

    Dim a As Double, b As Double, m As Double
    a = LO: b = HI
    For i = 1 To 200
        m = 0.5 * (a + b)
        diff = BS_PRICE(S, K, T, r, m, optType, q) - price
        If Abs(diff) < TOL Or (b - a) < 0.0000000001 Then Exit For
        If diff > 0 Then b = m Else a = m
    Next i
    BS_IMPLIED_VOL = m
    Exit Function
NoSolution:
    BS_IMPLIED_VOL = CVErr(xlErrNum)
    Exit Function
Fail:
    BS_IMPLIED_VOL = CVErr(xlErrValue)
End Function

'==============================================================================
' VaR macro
'   Inputs  (sheet "VaR Macro"): weights C5:C9 (stock names in B5:B9),
'           window C12, portfolio value C13, confidence levels C14 and C15
'   Data    (sheet "Returns"):   dates in column A, one column of daily simple
'           returns per stock, headers in row 1
'   Output  (sheet "VaR Macro"): table A20:G23, histogram data N19:O*, charts
'
'   Historical VaR  = -PERCENTILE.INC(portfolio returns, 1 - c)
'   Historical ES   = -average of the returns at or below that percentile
'   Parametric VaR  = z_c * stdev - mean        (z_c = NORM.S.INV(c))
'   Parametric ES   = stdev * phi(z_c) / (1 - c) - mean
'   The stdev of the portfolio return series equals sqrt(w' * Sigma * w).
'==============================================================================
Public Sub RefreshVaR()
    Dim wsIn As Worksheet, wsR As Worksheet
    On Error GoTo Fail
    Set wsIn = ThisWorkbook.Worksheets("VaR Macro")
    Set wsR = ThisWorkbook.Worksheets("Returns")

    '--- read inputs ---------------------------------------------------------
    Const N_STOCKS As Long = 5
    Dim names(1 To N_STOCKS) As String, w(1 To N_STOCKS) As Double, col(1 To N_STOCKS) As Long
    Dim i As Long, c As Long, t As Long, wSum As Double
    For i = 1 To N_STOCKS
        names(i) = CStr(wsIn.Cells(4 + i, 2).Value)
        w(i) = CDbl(wsIn.Cells(4 + i, 3).Value)
        wSum = wSum + w(i)
    Next i
    If Abs(wSum - 1) > 0.0001 Then
        If MsgBox("Weights sum to " & Format(wSum, "0.0%") & ", not 100%. Continue anyway?", _
                  vbYesNo + vbExclamation, "MarketLens") = vbNo Then Exit Sub
    End If

    Dim lastRow As Long, lastCol As Long, nObs As Long, window As Long
    lastRow = wsR.Cells(wsR.Rows.Count, 1).End(xlUp).Row
    lastCol = wsR.Cells(1, wsR.Columns.Count).End(xlToLeft).Column
    nObs = lastRow - 1
    window = CLng(wsIn.Range("C12").Value)
    If window < 20 Or window > nObs Then
        MsgBox "Window must be between 20 and " & nObs & " days.", vbExclamation, "MarketLens"
        Exit Sub
    End If

    ' match each stock name to its column on the Returns sheet
    For i = 1 To N_STOCKS
        col(i) = 0
        For c = 2 To lastCol
            If CStr(wsR.Cells(1, c).Value) = names(i) Then col(i) = c
        Next c
        If col(i) = 0 Then
            MsgBox "Stock '" & names(i) & "' not found in row 1 of sheet Returns.", vbExclamation, "MarketLens"
            Exit Sub
        End If
    Next i

    '--- portfolio return for each of the last `window` days -----------------
    ' Read the block into an array once: much faster than cell-by-cell.
    Dim data As Variant, port() As Double
    data = wsR.Range(wsR.Cells(lastRow - window + 1, 1), wsR.Cells(lastRow, lastCol)).Value
    ReDim port(1 To window)
    For t = 1 To window
        port(t) = 0
        For i = 1 To N_STOCKS
            port(t) = port(t) + w(i) * CDbl(data(t, col(i)))
        Next i
    Next t

    Dim wf As WorksheetFunction, mu As Double, sd As Double, portValue As Double
    Set wf = Application.WorksheetFunction
    mu = wf.Average(port)
    sd = wf.StDev_S(port)
    portValue = CDbl(wsIn.Range("C13").Value)

    '--- VaR and ES at both confidence levels --------------------------------
    Dim k As Long, conf As Double, qtl As Double, z As Double
    Dim hVaR As Double, hES As Double, pVaR As Double, pES As Double
    Dim tailSum As Double, tailN As Long, outRow As Long, asOf As Variant
    asOf = wsR.Cells(lastRow, 1).Value

    wsIn.Range("A20:G23").ClearContents
    For k = 0 To 1
        conf = CDbl(wsIn.Cells(14 + k, 3).Value)
        If conf <= 0.5 Or conf >= 1 Then
            MsgBox "Confidence levels must be between 50% and 100%.", vbExclamation, "MarketLens"
            Exit Sub
        End If

        ' historical: read the loss quantile straight off the data
        qtl = wf.Percentile_Inc(port, 1 - conf)
        hVaR = -qtl
        tailSum = 0: tailN = 0
        For t = 1 To window
            If port(t) <= qtl Then tailSum = tailSum + port(t): tailN = tailN + 1
        Next t
        hES = -tailSum / tailN

        ' parametric (variance-covariance): assume normal returns
        z = wf.Norm_S_Inv(conf)
        pVaR = z * sd - mu
        pES = sd * NormPdf(z) / (1 - conf) - mu

        outRow = 20 + 2 * k
        WriteVaRRow wsIn, outRow, "Historical", conf, hVaR, hES, portValue, asOf
        WriteVaRRow wsIn, outRow + 1, "Parametric", conf, pVaR, pES, portValue, asOf
    Next k

    BuildHistogram wsIn, port, window
    BuildVaRChart wsIn
    wsIn.Range("A26").Value = "Last refreshed " & Format(Now, "yyyy-mm-dd hh:nn") & _
                              " using " & window & " days ending " & Format(asOf, "yyyy-mm-dd") & "."
    Exit Sub
Fail:
    MsgBox "RefreshVaR failed: " & Err.Description, vbCritical, "MarketLens"
End Sub

Private Sub WriteVaRRow(ws As Worksheet, ByVal r As Long, ByVal method As String, ByVal conf As Double, _
                        ByVal varPct As Double, ByVal esPct As Double, ByVal portValue As Double, _
                        ByVal asOf As Variant)
    ws.Cells(r, 1).Value = method
    ws.Cells(r, 2).Value = Format(conf, "0.0%")
    ws.Cells(r, 3).Value = varPct: ws.Cells(r, 3).NumberFormat = "0.000%"
    ws.Cells(r, 4).Value = esPct:  ws.Cells(r, 4).NumberFormat = "0.000%"
    ws.Cells(r, 5).Value = varPct * portValue: ws.Cells(r, 5).NumberFormat = """Rs ""#,##0"
    ws.Cells(r, 6).Value = esPct * portValue:  ws.Cells(r, 6).NumberFormat = """Rs ""#,##0"
    ws.Cells(r, 7).Value = asOf:           ws.Cells(r, 7).NumberFormat = "yyyy-mm-dd"
End Sub

'------------------------------------------------------------------------------
' Histogram of the portfolio's daily returns (30 equal-width bins) in N19:O49
'------------------------------------------------------------------------------
Private Sub BuildHistogram(ws As Worksheet, port() As Double, ByVal n As Long)
    Const N_BINS As Long = 30
    Dim lo As Double, hi As Double, width As Double, counts(1 To N_BINS) As Long
    Dim t As Long, b As Long
    lo = Application.WorksheetFunction.Min(port)
    hi = Application.WorksheetFunction.Max(port)
    width = (hi - lo) / N_BINS
    If width <= 0 Then Exit Sub
    For t = 1 To n
        b = Int((port(t) - lo) / width) + 1
        If b > N_BINS Then b = N_BINS          ' the maximum lands in the last bin
        counts(b) = counts(b) + 1
    Next t

    ws.Range("N19:O60").ClearContents
    ws.Range("N19").Value = "Bin midpoint"
    ws.Range("O19").Value = "Days"
    For b = 1 To N_BINS
        ws.Cells(19 + b, 14).Value = lo + (b - 0.5) * width
        ws.Cells(19 + b, 14).NumberFormat = "0.0%"
        ws.Cells(19 + b, 15).Value = counts(b)
    Next b

    Dim co As ChartObject
    On Error Resume Next
    ws.ChartObjects("ReturnHistogram").Delete
    On Error GoTo 0
    Set co = ws.ChartObjects.Add(Left:=ws.Range("I28").Left, Top:=ws.Range("I28").Top, _
                                 Width:=430, Height:=250)
    co.Name = "ReturnHistogram"
    With co.Chart
        .ChartType = xlColumnClustered
        Do While .SeriesCollection.Count > 0
            .SeriesCollection(1).Delete
        Loop
        With .SeriesCollection.NewSeries
            .Name = "Days"
            .Values = ws.Range(ws.Cells(20, 15), ws.Cells(19 + N_BINS, 15))
            .XValues = ws.Range(ws.Cells(20, 14), ws.Cells(19 + N_BINS, 14))
        End With
        .ChartGroups(1).GapWidth = 10
        .HasTitle = True
        .ChartTitle.Text = "Portfolio daily returns (last " & n & " days)"
        .HasLegend = False
        .Axes(xlCategory).TickLabels.NumberFormat = "0.0%"
        .Axes(xlValue).HasTitle = True
        .Axes(xlValue).AxisTitle.Text = "Number of days"
    End With
End Sub

'------------------------------------------------------------------------------
' Clustered column chart of VaR and ES in rupees, from the results table
'------------------------------------------------------------------------------
Private Sub BuildVaRChart(ws As Worksheet)
    Dim co As ChartObject
    On Error Resume Next
    ws.ChartObjects("VaRChart").Delete
    On Error GoTo 0
    Set co = ws.ChartObjects.Add(Left:=ws.Range("A28").Left, Top:=ws.Range("A28").Top, _
                                 Width:=430, Height:=250)
    co.Name = "VaRChart"
    With co.Chart
        .ChartType = xlColumnClustered
        Do While .SeriesCollection.Count > 0
            .SeriesCollection(1).Delete
        Loop
        With .SeriesCollection.NewSeries
            .Name = "VaR (Rs)"
            .Values = ws.Range("E20:E23")
            .XValues = ws.Range("A20:B23")
        End With
        With .SeriesCollection.NewSeries
            .Name = "Expected Shortfall (Rs)"
            .Values = ws.Range("F20:F23")
            .XValues = ws.Range("A20:B23")
        End With
        .HasTitle = True
        .ChartTitle.Text = "1-day VaR and Expected Shortfall"
        .HasLegend = True
        .Legend.Position = xlLegendPositionBottom
        .Axes(xlValue).HasTitle = True
        .Axes(xlValue).AxisTitle.Text = "Rs"
        .Axes(xlValue).TickLabels.NumberFormat = "#,##0"
    End With
End Sub

'------------------------------------------------------------------------------
' One-off helper: add a button that runs RefreshVaR
'------------------------------------------------------------------------------
Public Sub AddRefreshButton()
    Dim ws As Worksheet, btn As Button
    Set ws = ThisWorkbook.Worksheets("VaR Macro")
    On Error Resume Next
    ws.Buttons("btnRefreshVaR").Delete
    On Error GoTo 0
    Set btn = ws.Buttons.Add(ws.Range("E12").Left, ws.Range("E12").Top, 130, 30)
    btn.Name = "btnRefreshVaR"
    btn.Caption = "Refresh VaR"
    btn.OnAction = "RefreshVaR"
End Sub
