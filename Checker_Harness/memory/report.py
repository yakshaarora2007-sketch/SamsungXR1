"""
memory/report.py - Renders memory constraint reports and tables according to official 8 MB heap/algorithm guidelines.
"""

from typing import Dict, List, Optional, Any
from .monitor import MemoryProfile, MemoryConstraintStatus, MemoryCoverage, MEMORY_LIMIT_BYTES


class MemoryReporter:
    """Generates Markdown and structured summaries for algorithm-level memory compliance."""

    @staticmethod
    def render_constraint_table() -> str:
        """Returns the official constraint breakdown table."""
        return """| Metric | Requirement | Status |
|---|---|---|
| Algorithm-level peak memory | <= 8 MB | Monitored & Enforced |
| Python interpreter baseline | Excluded | Exempt from 8 MB Limit |
| OS-level memory | Excluded | Exempt from 8 MB Limit |
| Native algorithm buffers | Included | Tracked in Algorithm Memory |
| Temporary algorithm allocations | Included | Tracked in Algorithm Memory |"""

    @staticmethod
    def render_markdown_section(profile: MemoryProfile) -> str:
        """Renders the memory analysis section for experiment.md or summary reports."""
        peak_b = profile.algorithm_peak_bytes
        peak_mb = profile.algorithm_peak_mb
        base_b = profile.interpreter_baseline_bytes
        base_mb = profile.interpreter_baseline_mb
        rss_peak_mb = profile.process_rss_peak_mb

        status_badge = profile.status.value
        coverage_badge = profile.coverage.value

        violation_text = ""
        if profile.status == MemoryConstraintStatus.VIOLATION and profile.violation_details:
            v = profile.violation_details
            violation_text = f"\n> [!WARNING] **MEMORY VIOLATION:** Operation exceeded 8 MB limit by {v['excess_bytes']:,} bytes ({v['excess_mb']} MB)!\n"

        buffer_lines = []
        if profile.tracked_buffers:
            buffer_lines.append("\n**Key Algorithm Buffers Tracked:**")
            for b_name, b_bytes in profile.tracked_buffers.items():
                b_kb = b_bytes / 1024.0
                buffer_lines.append(f"- `{b_name}`: {b_bytes:,} bytes ({b_kb:.2f} KB)")
        buffer_section = "\n".join(buffer_lines) if buffer_lines else ""

        return f"""## Memory Constraint Analysis (Algorithm / Heap Level)

{MemoryReporter.render_constraint_table()}

### Memory Metrics

- **Algorithm Peak Memory:** {peak_b:,} bytes ({peak_mb:.3f} MB) / 8,388,608 bytes (8.000 MB)
- **Constraint Status:** **{status_badge}**
- **Measurement Coverage:** {coverage_badge}
- **Python Interpreter Baseline (Excluded):** {base_b:,} bytes ({base_mb:.2f} MB)
- **Process RSS Peak (Supplementary Diagnostic):** {rss_peak_mb:.2f} MB
- **Measurement Methodology:** {profile.methodology}
{violation_text}{buffer_section}
"""

    @staticmethod
    def summarize_profiles(profiles: List[MemoryProfile]) -> Dict[str, Any]:
        """Summarizes multiple run memory profiles."""
        if not profiles:
            return {
                "max_peak_bytes": 0,
                "max_peak_mb": 0.0,
                "all_passed": True,
                "status": "PASS",
            }

        max_profile = max(profiles, key=lambda p: p.algorithm_peak_bytes)
        violations = [p for p in profiles if p.status == MemoryConstraintStatus.VIOLATION]

        return {
            "max_algorithm_peak_bytes": max_profile.algorithm_peak_bytes,
            "max_algorithm_peak_mb": round(max_profile.algorithm_peak_mb, 3),
            "interpreter_baseline_mb": round(max_profile.interpreter_baseline_mb, 3),
            "total_runs_monitored": len(profiles),
            "violations_count": len(violations),
            "all_passed": len(violations) == 0,
            "overall_status": "PASS" if len(violations) == 0 else "VIOLATION",
            "coverage": "FULL" if all(p.coverage == MemoryCoverage.FULL for p in profiles) else "PARTIALLY_VERIFIED",
        }
