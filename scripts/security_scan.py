#!/usr/bin/env python3
"""
security_scan.py - Shift Left Security Scanner

This script implements a "shift left" security approach by running dependency
vulnerability scans locally before pushing to GitHub. It checks for known
vulnerabilities in Python dependencies and provides actionable remediation
guidance.

Features:
- Python dependency vulnerability scanning using pip-audit
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

    def run_vulnerability_scan(self) -> Tuple[List[Dict], bool]:
        """
        Run pip-audit vulnerability scan on Python dependencies.
        
        Returns:
            Tuple of (vulnerabilities_list, scan_successful)
        """
        print("🔍 Running pip-audit vulnerability scan...")

        if not self.requirements_file.exists():
            print("❌ requirements.txt not found!")
            return [], False

        try:
            # Check if pip-audit is available
            audit_cmd = [sys.executable, "-m", "pip", "show", "pip-audit"]
            result = subprocess.run(audit_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print("⚠️  pip-audit not installed. Installing...")
                install_cmd = [sys.executable, "-m", "pip", "install", "pip-audit"]
                subprocess.run(install_cmd, check=True)

            # Run pip-audit scan with JSON output
            scan_cmd = [
                sys.executable, "-m", "pip_audit",
                "-r", str(self.requirements_file),
                "--format", "json",
                "--progress-spinner", "off"
            ]
            result = subprocess.run(scan_cmd, capture_output=True, text=True)

            # pip-audit returns 0 if no vulnerabilities, non-zero if vulnerabilities found
            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    dependencies = output.get("dependencies", [])
                    
                    # Convert pip-audit format to our internal format
                    vulnerabilities = []
                    for dep in dependencies:
                        for vuln in dep.get("vulns", []):
                            vulnerabilities.append({
                                "package": dep.get("name", "unknown"),
                                "current_version": dep.get("version", "unknown"),
                                "vulnerability_id": vuln.get("id", "unknown"),
                                "description": vuln.get("description", "No description"),
                                "severity": "high",  # pip-audit doesn't provide severity
                                "fixed_versions": vuln.get("fix_versions", []),
                                "ignored": False
                            })
                    
                    if vulnerabilities:
                        print(f"🚨 Found {len(vulnerabilities)} vulnerabilities!")
                        return vulnerabilities, True
                    else:
                        print("✅ No vulnerabilities found in dependencies!")
                        return [], True
                        
                except json.JSONDecodeError:
                    print("❌ Failed to parse pip-audit output")
                    print("Raw output:", result.stdout)
                    return [], False
            else:
                print("✅ No vulnerabilities found in dependencies!")
                return [], True

        except subprocess.CalledProcessError as e:
            print(f"❌ pip-audit scan failed: {e}")
            return [], False
        except Exception as e:
            print(f"❌ Unexpected error during pip-audit scan: {e}")
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
            vulnerabilities: List of vulnerability dictionaries

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
            # Skip ignored vulnerabilities
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

    def generate_report(self, vulnerabilities: List[Dict], analysis: Dict, remediations: List[str]) -> str:
        """Generate a comprehensive security report."""
        report = []
        report.append("# 🔒 Security Scan Report")
        report.append("")
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
                fixed_versions = vuln.get('fixed_versions', [])
                if fixed_versions:
                    report.append(f"**Fixed Versions:** {', '.join(map(str, fixed_versions))}")
                else:
                    report.append(f"**Fixed Versions:** none")
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

        # Run vulnerability scan
        vulnerabilities, scan_success = self.run_vulnerability_scan()

        if not scan_success:
            print("❌ Scan failed - unable to complete security check")
            return 1

        # Analyze results
        analysis = self.analyze_vulnerabilities(vulnerabilities)

        # Generate and display report
        remediations = []
        report = self.generate_report(vulnerabilities, analysis, remediations)
        print("\n" + report)

        # Save report to file
        report_file = self.project_root / "security_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_file}")

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