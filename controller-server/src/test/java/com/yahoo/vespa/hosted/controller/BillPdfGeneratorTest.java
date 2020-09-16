package com.yahoo.vespa.hosted.controller;

import com.yahoo.vespa.hosted.controller.api.integration.billing.Invoice;
import org.junit.Test;

import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class BillPdfGeneratorTest {

    @Test
    public void test() throws Exception {
        ZonedDateTime start = LocalDate.of(2020, 5, 23).atStartOfDay(ZoneId.systemDefault());
        ZonedDateTime end = start.plusDays(5);
        var statusHistory = new Invoice.StatusHistory(new TreeMap<>(Map.of(start, "OPEN")));
        List<Invoice.LineItem> items = List.of(
                lineItem("basic-test.aws (prod.aws-us-east-1c)", "475.72"),
                lineItem("album-recommendation.default-t (test.aws-us-east-1c)", "37.31"),
                lineItem("album-recommendation.default (test.aws-us-east-1c)", "51.27"),
                lineItem("album-recommendation.default (prod.aws-us-east-1c)", "1126.44"),
                lineItem("album-recommendation.default-t (staging.aws-us-east-1c)", "49.27"),
                lineItem("basic-test.musum (dev.aws-us-east-1c)", "0.11"),
                lineItem("album-recommendation.default (staging.aws-us-east-1c)", "135.00"),
                lineItem("restart.aws (prod.aws-us-east-1c)", "526.04"),
                lineItem("basic-test-global.default (prod.aws-us-east-1c)", "370.65"));
        Invoice invoice = new Invoice(Invoice.Id.of("id-1"), statusHistory, items, start, end);

        Files.write(Paths.get("test.pdf"), BillPdfGenerator.generate(invoice));
    }

    private static Invoice.LineItem lineItem(String description, String amount) {
        return new Invoice.LineItem("id", description, new BigDecimal(amount), "plan", "agent", LocalDate.of(2020, 5, 23).atStartOfDay(ZoneId.systemDefault()));
    }
}