#!/usr/bin/env python3
"""
security_scan.py - Shift Left Security Scanner

This script implements a "shift left" security approach by running dependency
vulnerability scans locally before pushing to GitHub. It checks for known
vulnerabilities in Python dependencies and provides actionable remediation
guidance.

Features:
- Python dependency vulnerability scanning using Safety
- GitHub Dependabot API integration for additional insights
- Pre-commit hook integration
- CI/CD pipeline integration
- Detailed reporting with severity levels
- Remediation suggestions

Usage:
    python scripts/security_scan.py [--fix] [--ci] [--strict]

Options:
    --fix       Attempt automatic remediation of fixable vulnerabilities
    --ci        CI mode - exit with error code on vulnerabilities
    --strict    Strict mode - fail on any vulnerabilities (not just high/critical)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

class SecurityScanner:
    """Main security scanner class implementing shift-left security practices."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.venv_path = project_root / "ml_env"
        self.requirements_file = project_root / "requirements.txt"
        self.results = {
            "vulnerabilities": [],
            "remediations": [],
            "summary": {}
        }

    def run_safety_scan(self) -> Tuple[List[Dict], bool]:
        """
        Run Safety vulnerability scan on Python dependencies.

        Returns:
            Tuple of (vulnerabilities_list, scan_successful)
        """
        print("🔍 Running Safety vulnerability scan...")

        if not self.requirements_file.exists():
            print("❌ requirements.txt not found!")
            return [], False

        try:
            # Check if safety is available in the environment
            safety_cmd = [sys.executable, "-m", "pip", "show", "safety"]
            result = subprocess.run(safety_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print("⚠️  Safety not installed. Installing...")
                install_cmd = [sys.executable, "-m", "pip", "install", "safety"]
                subprocess.run(install_cmd, check=True)

            # Run safety scan
            scan_cmd = [sys.executable, "-m", "safety", "scan", "--file", str(self.requirements_file), "--json"]
            result = subprocess.run(scan_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # No vulnerabilities found
                print("✅ No vulnerabilities found in dependencies!")
                return [], True
            else:
                # Parse JSON output for vulnerabilities
                try:
                    vulnerabilities = json.loads(result.stdout)
                    print(f"🚨 Found {len(vulnerabilities)} vulnerabilities!")
                    return vulnerabilities, True
                except json.JSONDecodeError:
                    print("❌ Failed to parse Safety output")
                    print("Raw output:", result.stdout)
                    return [], False

        except subprocess.CalledProcessError as e:
            print(f"❌ Safety scan failed: {e}")
            return [], False
        except Exception as e:
            print(f"❌ Unexpected error during safety scan: {e}")
            return [], False

    def check_github_dependabot(self) -> Optional[Dict]:
        """
        Check GitHub Dependabot status for the repository.

        Returns:
            Dependabot configuration if available, None otherwise
        """
        print("🔍 Checking GitHub Dependabot configuration...")

        dependabot_config = self.project_root / ".github" / "dependabot.yml"
        if dependabot_config.exists():
            print("✅ Dependabot configuration found")
            try:
                with open(dependabot_config, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    return config
            except ImportError:
                print("⚠️  PyYAML not available for parsing dependabot config")
            except Exception as e:
                print(f"❌ Error reading dependabot config: {e}")

        print("ℹ️  No Dependabot configuration found")
        return None

    def analyze_vulnerabilities(self, vulnerabilities: List[Dict]) -> Dict:
        """
        Analyze vulnerabilities and provide remediation guidance.

        Args:
            vulnerabilities: List of vulnerability dictionaries from Safety

        Returns:
            Analysis results with severity breakdown and recommendations
        """
        analysis = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "total": len(vulnerabilities),
            "total_unignored": 0,
            "fixable": [],
            "recommendations": []
        }

        severity_map = {
            "critical": analysis["critical"],
            "high": analysis["high"],
            "medium": analysis["medium"],
            "low": analysis["low"]
        }

        for vuln in vulnerabilities:
            # Skip ignored vulnerabilities (e.g., due to unpinned requirements)
            if vuln.get("ignored", False):
                continue

            analysis["total_unignored"] += 1

            severity = vuln.get("severity", "unknown").lower()
            if severity in severity_map:
                severity_map[severity].append(vuln)

            # Check if fixable (has a fix version available)
            if vuln.get("fixed_versions"):
                analysis["fixable"].append(vuln)

        # Generate recommendations
        if analysis["critical"]:
            analysis["recommendations"].append("🚨 CRITICAL: Immediate action required for critical vulnerabilities!")
        if analysis["high"]:
            analysis["recommendations"].append("⚠️  HIGH: Review and fix high-severity vulnerabilities before deployment")
        if analysis["fixable"]:
            analysis["recommendations"].append(f"🔧 {len(analysis['fixable'])} vulnerabilities have available fixes - run with --fix to attempt automatic remediation")

        return analysis

    def attempt_remediation(self, vulnerabilities: List[Dict]) -> List[str]:
        """
        Attempt automatic remediation of fixable vulnerabilities.

        Args:
            vulnerabilities: List of vulnerabilities to attempt to fix

        Returns:
            List of remediation actions taken
        """
        print("🔧 Attempting automatic remediation...")

        remediations = []
        fixable_vulns = [v for v in vulnerabilities if v.get("fixed_versions")]

        if not fixable_vulns:
            print("ℹ️  No automatically fixable vulnerabilities found")
            return remediations

        # Create backup of requirements.txt
        backup_file = self.requirements_file.with_suffix('.txt.backup')
        if self.requirements_file.exists():
            import shutil
            shutil.copy2(self.requirements_file, backup_file)
            remediations.append(f"📋 Created backup: {backup_file}")

        # For each fixable vulnerability, try to update the version
        for vuln in fixable_vulns:
            package = vuln.get("package", "")
            fixed_versions = vuln.get("fixed_versions", [])

            if not package or not fixed_versions:
                continue

            # Get the latest fixed version
            latest_fix = fixed_versions[-1] if isinstance(fixed_versions, list) else str(fixed_versions)

            print(f"🔄 Attempting to update {package} to {latest_fix}")

            try:
                # Use pip-tools or similar to update the requirement
                update_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", f"{package}=={latest_fix}"]
                result = subprocess.run(update_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    remediations.append(f"✅ Updated {package} to {latest_fix}")
                else:
                    remediations.append(f"❌ Failed to update {package}: {result.stderr}")

            except Exception as e:
                remediations.append(f"❌ Error updating {package}: {e}")

        return remediations

    def generate_report(self, vulnerabilities: List[Dict], analysis: Dict, remediations: List[str]) -> str:
        """Generate a comprehensive security report."""
        report = []
        report.append("# 🔒 Security Scan Report")
        report.append("")
        report.append(f"**Scan Date:** {os.popen('date').read().strip()}")
        report.append(f"**Project:** Smart-Secrets-Scanner")
        report.append(f"**Requirements File:** {self.requirements_file}")
        report.append("")

        # Summary
        report.append("## 📊 Summary")
        report.append("")
        report.append(f"- **Total Vulnerabilities Found:** {analysis['total']}")
        report.append(f"- **Unignored Vulnerabilities:** {analysis.get('total_unignored', 0)}")
        report.append(f"- **Critical:** {len(analysis['critical'])}")
        report.append(f"- **High:** {len(analysis['high'])}")
        report.append(f"- **Medium:** {len(analysis['medium'])}")
        report.append(f"- **Low:** {len(analysis['low'])}")
        report.append(f"- **Fixable:** {len(analysis['fixable'])}")
        report.append("")

        # Recommendations
        if analysis["recommendations"]:
            report.append("## 💡 Recommendations")
            report.append("")
            for rec in analysis["recommendations"]:
                report.append(f"- {rec}")
            report.append("")

        # Detailed vulnerabilities
        if vulnerabilities:
            report.append("## 🚨 Vulnerabilities Details")
            report.append("")

            for vuln in vulnerabilities:
                severity = vuln.get("severity", "unknown").upper()
                package = vuln.get("package", "unknown")
                vuln_id = vuln.get("vulnerability_id", "unknown")
                description = vuln.get("description", "No description available")

                report.append(f"### {severity}: {package} - {vuln_id}")
                report.append(f"**Description:** {description}")
                report.append(f"**Current Version:** {vuln.get('current_version', 'unknown')}")
                report.append(f"**Fixed Versions:** {', '.join(vuln.get('fixed_versions', ['none']))}")
                report.append("")

        # Remediations
        if remediations:
            report.append("## 🔧 Remediation Actions")
            report.append("")
            for remediation in remediations:
                report.append(f"- {remediation}")
            report.append("")

        return "\n".join(report)

    def run_scan(self, fix: bool = False, ci_mode: bool = False, strict: bool = False) -> int:
        """
        Run the complete security scan.

        Args:
            fix: Whether to attempt automatic remediation
            ci_mode: Whether running in CI (exit with error on vulnerabilities)
            strict: Whether to fail on any vulnerabilities (not just high/critical)

        Returns:
            Exit code (0 for success, 1 for vulnerabilities found)
        """
        print("🚀 Starting Shift-Left Security Scan")
        print("=" * 50)

        # Check Dependabot configuration
        dependabot_config = self.check_github_dependabot()

        # Run Safety scan
        vulnerabilities, scan_success = self.run_safety_scan()

        if not scan_success:
            print("❌ Scan failed - unable to complete security check")
            return 1

        # Analyze results
        analysis = self.analyze_vulnerabilities(vulnerabilities)

        # Attempt remediation if requested
        remediations = []
        if fix and vulnerabilities:
            remediations = self.attempt_remediation(vulnerabilities)

        # Generate and display report
        report = self.generate_report(vulnerabilities, analysis, remediations)
        print("\n" + report)

        # Determine exit code based on mode
        has_vulnerabilities = analysis.get("total_unignored", 0) > 0
        has_critical_high = len(analysis["critical"]) > 0 or len(analysis["high"]) > 0

        if ci_mode or strict:
            if strict and has_vulnerabilities:
                print("❌ STRICT MODE: Vulnerabilities found - failing build")
                return 1
            elif ci_mode and has_critical_high:
                print("❌ CI MODE: Critical/High vulnerabilities found - failing build")
                return 1

        if has_vulnerabilities:
            print("⚠️  Vulnerabilities found - review report above")
            return 1

        print("✅ Security scan passed - no vulnerabilities found")
        return 0


def main():
    """Main entry point for the security scanner."""
    parser = argparse.ArgumentParser(
        description="Shift-Left Security Scanner for Smart-Secrets-Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/security_scan.py                    # Basic scan
  python scripts/security_scan.py --fix             # Scan and attempt fixes
  python scripts/security_scan.py --ci              # CI mode (fail on high/critical)
  python scripts/security_scan.py --strict          # Strict mode (fail on any vulnerability)
        """
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt automatic remediation of fixable vulnerabilities"
    )

    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode - exit with error code on vulnerabilities"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode - fail on any vulnerabilities (not just high/critical)"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Initialize scanner
    scanner = SecurityScanner(project_root)

    # Run scan
    exit_code = scanner.run_scan(fix=args.fix, ci_mode=args.ci, strict=args.strict)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()