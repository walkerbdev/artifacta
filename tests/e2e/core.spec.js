import { test, expect } from '@playwright/test';

/**
 * Core E2E Tests for Artifacta UI
 *
 * These tests verify basic functionality of the web UI based on actual UI structure
 */

test.describe('Artifacta Core UI', () => {
  test('homepage loads successfully', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Artifacta/);
    await page.waitForLoadState('networkidle');
    await expect(page.locator('text=Projects')).toBeVisible();
    await expect(page.locator('text=Runs')).toBeVisible();
  });

  test('run list displays correctly', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Expand the Runs section
    await page.locator('text=Runs').first().click();
    await page.waitForTimeout(1000);

    // Verify run data appears
    await expect(page.locator('text=/all-primitives|Run/i').first()).toBeVisible({ timeout: 5000 });
  });

  test('navigation between tabs works', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.click('text=Notebooks');
    await page.click('text=Plots');
    await page.click('text=Tables');
    
    // Verify page is still responsive
    await expect(page.locator('text=Projects')).toBeVisible();
  });

  test('health check endpoint returns healthy', async ({ request }) => {
    const baseURL = process.env.ARTIFACTA_URL || 'http://localhost:8000';
    const response = await request.get(`${baseURL}/health`);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.status).toBe('healthy');
    expect(data.database_connected).toBe(true);
  });

  test('API returns run data', async ({ request }) => {
    const baseURL = process.env.ARTIFACTA_URL || 'http://localhost:8000';
    const response = await request.get(`${baseURL}/api/runs?limit=100`);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(Array.isArray(data)).toBeTruthy();
    expect(data.length).toBeGreaterThan(0);
    expect(data[0]).toHaveProperty('run_id');
    expect(data[0]).toHaveProperty('name');
  });

  test('sidebar is interactive', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await expect(page.locator('text=Projects').first()).toBeVisible();
    await expect(page.locator('text=Runs').first()).toBeVisible();
    await expect(page.locator('text=Files').first()).toBeVisible();
  });
});
