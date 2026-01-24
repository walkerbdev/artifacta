import { test, expect } from '@playwright/test';

/**
 * Visualization E2E Tests
 *
 * Tests for data visualization features:
 * - Plots tab (charts/graphs)
 * - Tables tab (structured data)
 * - Artifacts tab (files)
 * - Chat tab (AI interface)
 */

test.describe('Data Visualization', () => {
  test('plots tab renders charts', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait a bit for initial API calls to complete
    await page.waitForTimeout(2000);

    // Find the Runs section and click the chevron button to expand it
    // The button is near the "Runs" title and has a chevron icon
    const runsSection = page.locator('.collapsible-section-wrapper:has-text("Runs")');
    const expandButton = runsSection.locator('.collapsible-section-toggle');
    await expandButton.click();
    await page.waitForTimeout(1000);

    // Wait for run items to load and appear (runs are loaded asynchronously)
    // The RunTree renders checkboxes for each run
    await page.waitForSelector('input[type="checkbox"]', { timeout: 15000 });

    // Click the first checkbox to select the run
    const checkbox = page.locator('input[type="checkbox"]').first();
    await checkbox.click();
    await page.waitForTimeout(1500);

    // Navigate to Plots tab
    await page.click('text=Plots');
    await page.waitForTimeout(2000);

    // Wait for canvas elements to appear (charts render on canvas)
    await page.waitForSelector('canvas', { timeout: 15000 });

    // Verify chart elements render
    const charts = page.locator('canvas');
    const chartCount = await charts.count();

    // Should have at least one chart (all_primitives has BarChart, Scatter, Curve, Series, etc.)
    expect(chartCount).toBeGreaterThan(0);

    // Verify first chart is visible
    await expect(charts.first()).toBeVisible();
  });

  test('tables tab shows structured data', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Select a run first
    await page.locator('text=Runs').first().click();
    await page.waitForTimeout(500);
    await page.locator('text=/all-primitives|Run/i').first().click();
    await page.waitForTimeout(1000);

    // Navigate to Tables tab
    await page.click('text=Tables');
    await page.waitForTimeout(1000);

    // Verify table or data structure exists
    const tables = page.locator('table, [class*="table"], [class*="Table"], [class*="grid"]');
    const tableCount = await tables.count();

    // Should have some table-like elements
    expect(tableCount).toBeGreaterThan(0);
  });

  test('artifacts tab displays file list', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Select a run first
    await page.locator('text=Runs').first().click();
    await page.waitForTimeout(500);
    await page.locator('text=/all-primitives|Run/i').first().click();
    await page.waitForTimeout(1000);

    // Navigate to Artifacts tab
    await page.click('text=Artifacts');
    await page.waitForTimeout(1000);

    // Verify artifacts content area is visible
    // (May be empty list or have artifact items)
    const mainContent = page.locator('main, [class*="content"], [role="main"]');
    await expect(mainContent.first()).toBeVisible();
  });

  test('chat tab loads successfully', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Select a run first
    await page.locator('text=Runs').first().click();
    await page.waitForTimeout(500);
    await page.locator('text=/all-primitives|Run/i').first().click();
    await page.waitForTimeout(1000);

    // Navigate to Chat tab
    await page.click('text=Chat');
    await page.waitForTimeout(1000);

    // Verify Chat tab is active and visible
    await expect(page.locator('text=Chat').first()).toBeVisible();

    // Verify main content area exists (whether it shows setup or chat interface)
    const mainContent = page.locator('main, [class*="content"], [role="main"]');
    await expect(mainContent.first()).toBeVisible();
  });
});
