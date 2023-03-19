package ru.nsu.fit.tpnn.lab1;

import ru.nsu.fit.tpnn.lab1.panel.FeaturePanel;
import ru.nsu.fit.tpnn.lab1.panel.matrix.EntropyPanel;
import ru.nsu.fit.tpnn.lab1.panel.matrix.MatrixPanel;
import ru.nsu.fit.tpnn.lab1.selection.Selection;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.io.File;
import java.math.BigDecimal;
import java.util.List;

public class MetricsFrame extends JFrame {

    private final JFileChooser fileChooser = new JFileChooser();
    private final DataReader dataReader = new DataReader();

    private List<Selection> selections;

    private FeaturePanel featurePanel;

    private JScrollPane matricesPanelScrollPane;

    public static void main(String args[]) {
        EventQueue.invokeLater(() -> new MetricsFrame().setVisible(true));
    }

    public MetricsFrame() {
        super();
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        Toolkit toolkit = Toolkit.getDefaultToolkit();
        Dimension screenSize = toolkit.getScreenSize();
        setPreferredSize(new Dimension((int) (9 * screenSize.getWidth() / 10),
                (int) (9 * screenSize.getHeight() / 10)));

        fileChooser.setCurrentDirectory(new File(".."));
        fileChooser.setMultiSelectionEnabled(false);
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        fileChooser.setAcceptAllFileFilterUsed(false);
        fileChooser.addChoosableFileFilter(new FileNameExtensionFilter("xlsx", "xlsx"));

        JPanel filePanel = new JPanel();
        addTextFieldAndLabels(filePanel);

        JButton chooseFileButton = new JButton("choose file");
        chooseFileButton.addActionListener(e -> {
            int result = fileChooser.showOpenDialog(MetricsFrame.this);
            if (JFileChooser.APPROVE_OPTION == result) {
                try {
                    int namesRowIndex = Integer.parseInt(namesRowIndexField.getText());
                    int unitsRowIndex = Integer.parseInt(unitsRowIndexField.getText());
                    int firstDataRowIndex = Integer.parseInt(firstDataRowIndexField.getText());
                    int lastDataRowIndex = Integer.parseInt(lastDataRowIndexField.getText());
                    int lastColumnIndex = Integer.parseInt(lastColumnIndexField.getText());
                    selections =
                            dataReader.readFile(fileChooser.getSelectedFile(),
                                    namesRowIndex - 1,
                                    unitsRowIndex - 1,
                                    firstDataRowIndex - 1,
                                    lastDataRowIndex - 1,
                                    lastColumnIndex - 1);
                    if (null != featurePanel) {
                        remove(featurePanel);
                    }
                    if (null != matricesPanelScrollPane) {
                        remove(matricesPanelScrollPane);
                    }
                    add(featurePanel = new FeaturePanel(selections), BorderLayout.CENTER);
                    revalidate();
                    pack();
                    setLocationRelativeTo(null);
                } catch (NumberFormatException ignored) {
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });
        filePanel.add(chooseFileButton);
        add(filePanel, BorderLayout.NORTH);

        JPanel findButtonPanel = new JPanel();
        JButton findButton = new JButton("find");
        findButton.addActionListener(e -> {
            if (null != featurePanel) {
                remove(featurePanel);
            } else {
                return;
            }
            if (null != matricesPanelScrollPane) {
                remove(matricesPanelScrollPane);
            }
            JPanel matricesPanel = new JPanel();
            matricesPanel.setLayout(new BoxLayout(matricesPanel, BoxLayout.Y_AXIS));

            EntropyPanel entropyNoDataPanel = new EntropyPanel(selections, true);
            EntropyPanel entropyPanel = new EntropyPanel(selections, false);
            MatrixPanel correlationPanel = new MatrixPanel(selections, selections,
                    getCorrelations(), "Correlation");
            MatrixPanel gainRatioNoDataPanel = new MatrixPanel(selections, featurePanel.getTargets(),
                    featurePanel.getGainRatios(true),
                    "Gain Ratio counting no data entries");
            MatrixPanel gainRatioPanel = new MatrixPanel(selections, featurePanel.getTargets(),
                    featurePanel.getGainRatios(false), "Gain Ratio");
            matricesPanel.add(entropyNoDataPanel);
            matricesPanel.add(entropyPanel);
            matricesPanel.add(correlationPanel);
            matricesPanel.add(gainRatioNoDataPanel);
            matricesPanel.add(gainRatioPanel);

            matricesPanelScrollPane = new JScrollPane(matricesPanel);
            add(matricesPanelScrollPane);
            revalidate();
        });
        findButtonPanel.add(findButton);
        add(findButtonPanel, BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(null);
    }

    private JTextField namesRowIndexField;
    private JTextField unitsRowIndexField;
    private JTextField firstDataRowIndexField;
    private JTextField lastDataRowIndexField;
    private JTextField lastColumnIndexField;

    private void addTextFieldAndLabels(JPanel filePanel) {
        namesRowIndexField = new JTextField("2", 2);
        unitsRowIndexField = new JTextField("3", 2);
        firstDataRowIndexField = new JTextField("4", 2);
        lastDataRowIndexField = new JTextField("188", 3);
        lastColumnIndexField = new JTextField("33", 2);
        JLabel namesRowIndexLabel = new JLabel("names row index");
        JLabel unitsRowIndexLabel = new JLabel("units row index");
        JLabel firstDataRowIndexLabel = new JLabel("first data row index");
        JLabel lastDataRowIndexLabel = new JLabel("last data row index");
        JLabel lastColumnIndexLabel = new JLabel("last column index");
        filePanel.add(namesRowIndexLabel);
        filePanel.add(namesRowIndexField);
        filePanel.add(unitsRowIndexLabel);
        filePanel.add(unitsRowIndexField);
        filePanel.add(firstDataRowIndexLabel);
        filePanel.add(firstDataRowIndexField);
        filePanel.add(lastDataRowIndexLabel);
        filePanel.add(lastDataRowIndexField);
        filePanel.add(lastColumnIndexLabel);
        filePanel.add(lastColumnIndexField);
    }

    private BigDecimal[][] getCorrelations() {
        if (null == selections) {
            return null;
        }
        BigDecimal[][] correlations = new BigDecimal[selections.size()][selections.size()];
        for (int i = 0; i < selections.size(); ++i) {
            for (int j = i; j < selections.size(); ++j) {
                correlations[i][j] = selections.get(i).getCorrelation(selections.get(j));
            }
            for (int j = 0; j < i; ++j) {
                correlations[i][j] = BigDecimal.valueOf(0);
            }
        }
        return correlations;
    }

}
