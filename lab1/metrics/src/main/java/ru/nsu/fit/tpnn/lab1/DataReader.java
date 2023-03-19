package ru.nsu.fit.tpnn.lab1;

import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.openxml4j.opc.OPCPackage;
import org.apache.poi.openxml4j.util.ZipSecureFile;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import ru.nsu.fit.tpnn.lab1.selection.Selection;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.poi.ss.usermodel.CellType.*;

public class DataReader {

    public List<Selection> readFile(File file, int parameterNamesRowIndex, int unitsNamesRowIndex,
                                    int firstDataRowInd, int lastRowInd, int lastColumnInd)
            throws InvalidFormatException, IOException {
        ZipSecureFile.setMinInflateRatio(0);
        XSSFWorkbook workbook = new XSSFWorkbook(OPCPackage.open(file));
        Sheet sheet = workbook.getSheetAt(0);

        final List<Selection> selections = new ArrayList<>();
        Row nameRow = sheet.getRow(parameterNamesRowIndex);
        Row unitRow = sheet.getRow(unitsNamesRowIndex);
        for (Cell cell : nameRow) {
            if (cell.getColumnIndex() > lastColumnInd) {
                break;
            }
            Cell unitCell = unitRow.getCell(cell.getColumnIndex());
            if (STRING == cell.getCellType() || STRING == unitCell.getCellType()) {
                selections.add(new Selection(cell.getRichStringCellValue().getString(),
                        unitCell.getRichStringCellValue().getString()));
            }
        }
        int entryInd;
        for (entryInd = firstDataRowInd; entryInd <= lastRowInd; ++entryInd) {
            Row row = sheet.getRow(entryInd);
            if (null == row || _NONE == row.getCell(0).getCellType()) {
                break;
            }
            for (Cell cell : row) {
                int columnIndex = cell.getColumnIndex();
                if (columnIndex > lastColumnInd) {
                    break;
                }
                CellType cellType = cell.getCellType();
                if (FORMULA == cellType || NUMERIC == cellType) {
                    Selection currentSelection = selections.get(columnIndex);
                    currentSelection.setNextValue(entryInd - firstDataRowInd,
                            cell.getNumericCellValue());
                }
            }
        }
        for (int i = 0; i < selections.size(); ++i) {
            Selection selection = selections.get(i);
            if (selection.isEmpty()) {
                selections.remove(selection);
            } else {
                selection.setNoDataEntries(entryInd - firstDataRowInd);
            }
        }
        return selections;
    }
}
