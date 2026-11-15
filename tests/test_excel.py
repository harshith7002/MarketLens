"""Tests for excel_export.py.  Run with:  pytest -q"""

import pytest
from openpyxl import load_workbook

from marketlens import excel_export


@pytest.fixture(scope="module")
def wb(tmp_path_factory):
    path = tmp_path_factory.mktemp("xl") / "MarketLens.xlsx"
    excel_export.build_workbook(path)
    return load_workbook(path)


def test_all_sheets_present(wb):
    assert wb.sheetnames == ["Summary", "Risk", "VaR", "VaR Macro", "Greeks", "Pricer", "Pairs", "Returns"]


def test_vba_cell_map_unchanged(wb):
    """MarketLens.bas reads these exact addresses - if this fails, update the .bas too."""
    ws = wb["VaR Macro"]
    assert [ws.cell(row=r, column=2).value for r in range(5, 10)] == \
        ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    assert sum(ws.cell(row=r, column=3).value for r in range(5, 10)) == pytest.approx(1.0)
    assert ws["C12"].value == 250
    assert ws["C14"].value == 0.95 and ws["C15"].value == 0.99
    assert ws["A19"].value == "Method"
    rs = wb["Returns"]
    assert rs["A1"].value == "Date" and rs["B1"].value == "RELIANCE"


def test_pricer_uses_vba_function_names(wb):
    ws = wb["Pricer"]
    assert ws["D16"].value.startswith("=BS_PRICE(")
    assert ws["D17"].value.startswith('=BS_GREEK("delta"')
    assert ws["D25"].value.startswith("=BS_IMPLIED_VOL(")
    assert "_xlfn.NORM.S.DIST" in ws["C16"].value


def test_var_ordering_in_workbook(wb):
    ws = wb["VaR"]
    rows = {(ws.cell(row=r, column=1).value, ws.cell(row=r, column=2).value): ws.cell(row=r, column=3).value
            for r in range(5, 11)}
    for m in ("Historical", "Parametric", "Monte Carlo"):
        assert rows[(m, "95%")] < rows[(m, "99%")]


def test_vba_file_exports_expected_procedures():
    from marketlens import config
    text = (config.ROOT_DIR / "vba" / "MarketLens.bas").read_bytes()
    assert b"\r\n" in text, "VBA import expects Windows (CRLF) line endings"
    src = text.decode("ascii")
    assert src.startswith('Attribute VB_Name = "MarketLens"')
    for proc in ("Public Function BS_PRICE", "Public Function BS_GREEK", "Public Function BS_IMPLIED_VOL",
                 "Public Sub RefreshVaR", "Public Sub AddRefreshButton"):
        assert proc in src
