// Parity tests: the Vite app must match the published artifact pixel for pixel.
// Each scenario runs the same script against both pages under a frozen clock,
// then compares full-page screenshots.
import { test, expect } from '@playwright/test';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

const APP = 'http://localhost:4173/';
const REF = 'http://reference.local/';
// Skeleton the artifact host wraps around every page (captured from the live artifact)
const WRAP = '<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}html{scroll-padding-top:env(safe-area-inset-top,0px)}body{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#faf9f5;color:#141413}img{max-width:100%}[hidden]:not([hidden=until-found i]){display:none!important}</style></head><body>';
const REF_HTML = WRAP + readFileSync(new URL('../../atlas/ecg_phase_atlas_artifact.html', import.meta.url), 'utf8') + '</body></html>';
const MAX_DIFF = 0.002; // fraction of pixels allowed to differ

async function open(page, url) {
  await page.route(REF, r => r.fulfill({ contentType: 'text/html', body: REF_HTML }));
  await page.clock.install({ time: new Date('2026-10-19T12:00:00Z') });
  await page.clock.pauseAt(new Date('2026-10-19T12:00:00Z'));
  await page.goto(url);
  await page.waitForFunction(() => document.getElementById('sub')?.textContent.length > 0);
  await page.evaluate(() => document.fonts.ready);
  await page.clock.runFor(1500);
}

async function setRange(page, id, value) {
  await page.evaluate(([id, v]) => {
    const el = document.getElementById(id); el.value = v; el.dispatchEvent(new Event('input', { bubbles: true }));
  }, [id, value]);
}

const scenarios = {
  'initial': async () => {},
  'method 6 morph done': async page => { await page.keyboard.press('6'); await page.clock.runFor(1000); },
  'mid-morph 2 to 5': async page => { await page.keyboard.press('2'); await page.clock.runFor(1000); await page.keyboard.press('5'); await page.clock.runFor(300); },
  'pause + card click + face-on': async page => {
    await page.keyboard.press(' ');
    await page.locator('.card').nth(3).click();
    await page.clock.runFor(1000);
    await page.locator('#face').click();
    await page.clock.runFor(1000);
  },
  'toggles + sliders': async page => {
    await page.locator('#allbeats').click();
    await page.locator('#rotate').click();
    await setRange(page, 'glow', '1.6');
    await setRange(page, 'trail', '0.7');
    await setRange(page, 'speed', '1.25');
    await page.clock.runFor(1200);
  },
  'scrub trace': async page => {
    await page.locator('#play').click();
    const b = await page.locator('#trace svg').boundingBox();
    await page.mouse.move(b.x + b.width * 0.3, b.y + b.height / 2);
    await page.mouse.down();
    await page.mouse.move(b.x + b.width * 0.62, b.y + b.height / 2, { steps: 8 });
    await page.mouse.up();
    await page.clock.runFor(100);
  },
  'recon tooltip': async page => {
    const b = await page.locator('#recon svg').boundingBox();
    await page.mouse.move(b.x + b.width * 0.2, b.y + b.height * 0.5);
    await page.clock.runFor(100);
  },
};

const viewports = { desktop: { width: 1440, height: 1000 }, phone: { width: 390, height: 844 } };

for (const [vpName, viewport] of Object.entries(viewports)) {
  for (const [name, run] of Object.entries(scenarios)) {
    if (vpName === 'phone' && !['initial', 'method 6 morph done', 'pause + card click + face-on'].includes(name)) continue;
    test(`${vpName}: ${name}`, async ({ browser }) => {
      const shots = [];
      for (const url of [REF, APP]) {
        const ctx = await browser.newContext({ viewport });
        const page = await ctx.newPage();
        await open(page, url);
        await run(page);
        shots.push(PNG.sync.read(await page.screenshot({ fullPage: true })));
        await ctx.close();
      }
      const [a, b] = shots;
      expect(`${b.width}x${b.height}`).toBe(`${a.width}x${a.height}`);
      const diff = new PNG({ width: a.width, height: a.height });
      const n = pixelmatch(a.data, b.data, diff.data, a.width, a.height, { threshold: 0.1 });
      const ratio = n / (a.width * a.height);
      if (ratio > 0) {
        const dir = new URL(`../test-results/parity/`, import.meta.url);
        mkdirSync(dir, { recursive: true });
        const base = `${vpName}-${name.replace(/\W+/g, '_')}`;
        writeFileSync(new URL(`${base}-ref.png`, dir), PNG.sync.write(a));
        writeFileSync(new URL(`${base}-app.png`, dir), PNG.sync.write(b));
        writeFileSync(new URL(`${base}-diff.png`, dir), PNG.sync.write(diff));
      }
      console.log(`  ${vpName}/${name}: ${n} px differ (${(ratio * 100).toFixed(4)}%)`);
      expect(ratio).toBeLessThanOrEqual(MAX_DIFF);
    });
  }
}
