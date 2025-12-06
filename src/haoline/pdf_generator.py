# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
PDF generation for HaoLine using Playwright.

This module provides PDF generation from HTML reports using Playwright,
which renders the HTML with a real browser engine for high-quality output.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .report import InspectionReport

# Check for Playwright availability
_HAS_PLAYWRIGHT = False
try:
    from playwright.async_api import async_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if Playwright is available for PDF generation."""
    return _HAS_PLAYWRIGHT


class PDFGenerator:
    """
    Generate PDF reports from HTML using Playwright.

    Playwright provides high-quality PDF rendering using Chromium,
    ensuring consistent output across platforms.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        page_format: str = "A4",
        landscape: bool = False,
        print_background: bool = True,
        margin_top: str = "20mm",
        margin_bottom: str = "20mm",
        margin_left: str = "15mm",
        margin_right: str = "15mm",
    ):
        """
        Initialize PDF generator.

        Args:
            logger: Logger instance
            page_format: Page format (A4, Letter, Legal, etc.)
            landscape: Use landscape orientation
            print_background: Include background colors/images
            margin_top: Top margin (CSS units)
            margin_bottom: Bottom margin (CSS units)
            margin_left: Left margin (CSS units)
            margin_right: Right margin (CSS units)
        """
        self.logger = logger or logging.getLogger("haoline.pdf")
        self.page_format = page_format
        self.landscape = landscape
        self.print_background = print_background
        self.margin = {
            "top": margin_top,
            "bottom": margin_bottom,
            "left": margin_left,
            "right": margin_right,
        }

    async def _generate_pdf_async(
        self,
        html_content: str,
        output_path: pathlib.Path,
    ) -> bool:
        """
        Async implementation of PDF generation.

        Args:
            html_content: HTML string to convert
            output_path: Path for output PDF

        Returns:
            True if successful, False otherwise
        """
        if not _HAS_PLAYWRIGHT:
            self.logger.error(
                "Playwright not installed. Install with: pip install playwright && playwright install chromium"
            )
            return False

        try:
            async with async_playwright() as p:
                # Launch headless Chromium
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                # Set the HTML content
                await page.set_content(html_content, wait_until="networkidle")

                # Add custom CSS for better PDF rendering with smart page breaks
                await page.add_style_tag(
                    content="""@media print {
body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
.no-print, button, .toggle-btn, .search-box { display: none !important; }
pre, code { white-space: pre-wrap !important; word-wrap: break-word !important; max-width: 100% !important; overflow-wrap: break-word !important; }
p, li { orphans: 3; widows: 3; }
h1, h2, h3, h4, h5, h6 { page-break-after: avoid !important; break-after: avoid !important; }
section { page-break-inside: avoid; break-inside: avoid; }
.kv-cache, .memory-breakdown, .visualizations, .graph-section, .layer-summary, .architecture, .hardware, .risks, .batch-scaling, .resolution-scaling { page-break-before: always !important; break-before: page !important; }
.executive-summary, .metrics-cards, .param-details, .dataset-info, .system-requirements { page-break-inside: avoid !important; break-inside: avoid !important; }
table { page-break-inside: avoid !important; break-inside: avoid !important; }
tr { page-break-inside: avoid !important; break-inside: avoid !important; }
figure, .chart-container, .visualization-item { page-break-inside: avoid !important; break-inside: avoid !important; }
img { page-break-inside: avoid !important; break-inside: avoid !important; max-width: 100% !important; height: auto !important; }
.metric-card, .card { page-break-inside: avoid !important; break-inside: avoid !important; }
.risk-item, .risk-signal { page-break-inside: avoid !important; break-inside: avoid !important; }
.comparison-table, .variant-table { page-break-inside: avoid !important; }
.engine-panel, .summary-panel { page-break-inside: avoid !important; break-inside: avoid !important; }
.recommendation, .calibration-rec { page-break-inside: avoid !important; break-inside: avoid !important; }
}"""
                )

                # Wait for any images to load
                await page.wait_for_load_state("networkidle")

                # Generate PDF
                await page.pdf(
                    path=str(output_path),
                    format=self.page_format,
                    landscape=self.landscape,
                    print_background=self.print_background,
                    margin=self.margin,  # type: ignore[arg-type]
                    display_header_footer=True,
                    header_template='<div style="font-size: 9px; color: #666; width: 100%; text-align: center; padding: 5px 0;">HaoLine Report</div>',
                    footer_template='<div style="font-size: 9px; color: #666; width: 100%; text-align: center; padding: 5px 0;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                )

                await browser.close()
                return True

        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            return False

    def generate_from_html(
        self,
        html_content: str,
        output_path: pathlib.Path,
    ) -> bool:
        """
        Generate PDF from HTML content.

        Args:
            html_content: HTML string to convert
            output_path: Path for output PDF

        Returns:
            True if successful, False otherwise
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating PDF: {output_path}")

        # Run async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._generate_pdf_async(html_content, output_path))

    def generate_from_html_file(
        self,
        html_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> bool:
        """
        Generate PDF from an HTML file.

        Args:
            html_path: Path to HTML file
            output_path: Path for output PDF

        Returns:
            True if successful, False otherwise
        """
        html_path = pathlib.Path(html_path)
        if not html_path.exists():
            self.logger.error(f"HTML file not found: {html_path}")
            return False

        html_content = html_path.read_text(encoding="utf-8")
        return self.generate_from_html(html_content, output_path)

    def generate_from_report(
        self,
        report: InspectionReport,
        output_path: pathlib.Path,
        image_paths: dict[str, pathlib.Path] | None = None,
    ) -> bool:
        """
        Generate PDF directly from an InspectionReport.

        Args:
            report: InspectionReport instance
            output_path: Path for output PDF
            image_paths: Optional dict of image paths for visualizations

        Returns:
            True if successful, False otherwise
        """
        # Generate HTML with embedded images (for PDF, all images are base64)
        html_content = report.to_html(image_paths=image_paths)
        return self.generate_from_html(html_content, output_path)


async def generate_pdf_async(
    html_content: str,
    output_path: pathlib.Path,
    **kwargs,
) -> bool:
    """
    Convenience async function for PDF generation.

    Args:
        html_content: HTML string to convert
        output_path: Path for output PDF
        **kwargs: Additional options for PDFGenerator

    Returns:
        True if successful, False otherwise
    """
    generator = PDFGenerator(**kwargs)
    return await generator._generate_pdf_async(html_content, output_path)


def generate_pdf(
    html_content: str,
    output_path: pathlib.Path,
    **kwargs,
) -> bool:
    """
    Convenience function for PDF generation.

    Args:
        html_content: HTML string to convert
        output_path: Path for output PDF
        **kwargs: Additional options for PDFGenerator

    Returns:
        True if successful, False otherwise
    """
    generator = PDFGenerator(**kwargs)
    return generator.generate_from_html(html_content, output_path)
