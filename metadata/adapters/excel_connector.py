"""
Adapter for ingesting Excel spreadsheet data sources.

Reads .xlsx and legacy .xls files using openpyxl / xlrd. Supports
multi-sheet extraction and merging, header row detection, and
stripping Excel formatting artefacts (merged cells, hidden rows).

Returns a clean pd.DataFrame tagged with sheet_name and source_file.

"""

