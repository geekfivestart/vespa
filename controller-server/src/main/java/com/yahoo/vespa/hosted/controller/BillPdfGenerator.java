package com.yahoo.vespa.hosted.controller;

import com.helger.pdflayout4.PageLayoutPDF;
import com.helger.pdflayout4.base.PLPageSet;
import com.helger.pdflayout4.element.box.PLBox;
import com.helger.pdflayout4.element.hbox.PLHBox;
import com.helger.pdflayout4.element.image.PLImage;
import com.helger.pdflayout4.element.table.EPLTableGridType;
import com.helger.pdflayout4.element.table.PLCellRange;
import com.helger.pdflayout4.element.table.PLTable;
import com.helger.pdflayout4.element.table.PLTableCell;
import com.helger.pdflayout4.element.table.PLTableRow;
import com.helger.pdflayout4.element.text.PLText;
import com.helger.pdflayout4.element.vbox.PLVBox;
import com.helger.pdflayout4.spec.BorderStyleSpec;
import com.helger.pdflayout4.spec.EHorzAlignment;
import com.helger.pdflayout4.spec.FontSpec;
import com.helger.pdflayout4.spec.PaddingSpec;
import com.helger.pdflayout4.spec.PreloadFont;
import com.helger.pdflayout4.spec.WidthSpec;
import com.yahoo.vespa.hosted.controller.api.integration.billing.Invoice;
import org.apache.pdfbox.pdmodel.common.PDRectangle;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.format.DateTimeFormatter;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class BillPdfGenerator {

    private static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final FontSpec r10 = new FontSpec(PreloadFont.REGULAR, 10);
    private static final FontSpec r10b = new FontSpec(PreloadFont.REGULAR_BOLD, 10);
    private static final FontSpec r14b = new FontSpec(PreloadFont.REGULAR_BOLD, 14);
    private static final PaddingSpec padding = new PaddingSpec(2);

    public static byte[] generate(Invoice invoice) throws Exception {
        PLPageSet pageSet = new PLPageSet(PDRectangle.A4).setPadding(25);

        BufferedImage vespaLogo = ImageIO.read(Files.newInputStream(Paths.get("VespaLogoBlack.png")));
        PLHBox logoInvoiceBox = new PLHBox();
        logoInvoiceBox.addColumn(new PLImage(vespaLogo, 125, 42), WidthSpec.star());
        logoInvoiceBox.addColumn(new PLText("INVOICE", r14b), WidthSpec.star());
        pageSet.addElement(logoInvoiceBox);


        PLHBox detailsBoxes = new PLHBox();
        addDetailsBox(detailsBoxes, 50, "Verizon Media", "Prinsens Gate 49", "7011 Trondheim", "Norway");
        addDetailsBox(detailsBoxes, 25, "Tenant:", "From:", "To:", "Support:");
        addDetailsBox(detailsBoxes, 25, "Yahoo! Japan", invoice.getStartTime().format(DATE_TIME_FORMATTER), invoice.getEndTime().format(DATE_TIME_FORMATTER), "https://vespa.zendesk.com");
        pageSet.addElement(new PLBox(detailsBoxes).setMargin(20, 0));

        PLBox paymentHeaderBox = new PLBox(new PLText("Payment Details", r14b))
                .setBorderBottom(new BorderStyleSpec(new Color(0xecf5fb), 1))
                .setPadding(5, 0)
                .setMargin(5, 0);
        pageSet.addElement(paymentHeaderBox);


        PLTable aTable = new PLTable(WidthSpec.star(), WidthSpec.abs(50), WidthSpec.abs(70), WidthSpec.abs(70));
        aTable.setHeaderRowCount(1);

        PLTableRow aHeaderRow = aTable.addAndReturnRow(
                new PLTableCell(new PLText("Line", r10b).setPadding(padding)),
                new PLTableCell(new PLText("Resource", r10b).setPadding(padding)),
                new PLTableCell(new PLText("Hours", r10b).setPadding(padding)).setHorzAlign(EHorzAlignment.RIGHT),
                new PLTableCell(new PLText("USD", r10b).setPadding(padding)).setHorzAlign(EHorzAlignment.RIGHT));
        aHeaderRow.setFillColor(new Color(0xd7d7d7)); // c9f2ff

        // Add content lines
        for (Invoice.LineItem lineItem : invoice.lineItems()) addLineItemRow(aTable, lineItem);
        for (Invoice.LineItem lineItem : invoice.lineItems()) addLineItemRow(aTable, lineItem);
        for (Invoice.LineItem lineItem : invoice.lineItems()) addLineItemRow(aTable, lineItem);

        EPLTableGridType.HORZ_ALL.applyGridToTable(aTable,
                new PLCellRange(2, aTable.getRowCount() - 2, 0, aTable.getColumnCount()),
                new BorderStyleSpec(new Color(0xe2e2e2), 1));
        pageSet.addElement(aTable);
        pageSet.addElement(new PLText("Text after table", r10));


        PageLayoutPDF aPageLayout = new PageLayoutPDF();
        aPageLayout.addPageSet(pageSet);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        aPageLayout.renderTo(baos);
        return baos.toByteArray();
    }

    private static void addSummaryRow(PLTable table, String heading, String value) {
        table.addRow(
                new PLTableCell(new PLText(heading, r10b)),
                new PLTableCell(new PLText(value, r10)).setHorzAlign(EHorzAlignment.RIGHT));
    }

    private static void addDetailsBox(PLHBox parent, int widthPercent, String heading, String... rest) {
        PLVBox box = new PLVBox();
        box.addRow(new PLText(heading, r10b));
        for (String field: rest) box.addRow(new PLText(field, r10));
        parent.addColumn(box, WidthSpec.perc(widthPercent));
    }

    private static void addLineItemRow(PLTable table, Invoice.LineItem lineItem) {
        int cpuHours = randInt(100, 10000);
        int memHours = (int) (cpuHours * (6 + 4 * Math.random()));
        int diskHours = (int) (cpuHours * (100 + 50 * Math.random()));
        String resourceHours = Stream.of(cpuHours, memHours, diskHours).map(v -> String.format("%,d", v)).collect(Collectors.joining("\n"));
        String cost = Stream.of(cpuHours * 0.12, memHours * 0.012, diskHours * 0.0004).map(v -> String.format("%,.2f", v)).collect(Collectors.joining("\n"));

        table.addRow(
                new PLTableCell(new PLText(lineItem.description(), r10).setPadding(padding)),
                new PLTableCell(new PLText("CPU\nRAM\nDisk", r10).setPadding(padding)),
                new PLTableCell(new PLText(resourceHours, r10).setPadding(padding).setHorzAlign(EHorzAlignment.RIGHT)).setHorzAlign(EHorzAlignment.RIGHT),
                new PLTableCell(new PLText(cost, r10).setPadding(padding).setHorzAlign(EHorzAlignment.RIGHT)).setHorzAlign(EHorzAlignment.RIGHT));
    }

    private static int randInt(int low, int high) {
        return (int) (low + (high - low) * Math.random());
    }
}
