# src/agents/investigation_agent.py
"""
HR Investigation Agent.
Uses tools to investigate employee attrition risk.
Demonstrates agentic AI pattern: tools + reasoning + action.
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def investigate_employee(employee_id: int) -> dict:
    """
    Run complete investigation for an employee.
    Calls tools in sequence: details → risk → department.

    Args:
        employee_id: Integer from 1 to 1470

    Returns:
        dict with complete investigation results
    """
    from src.agents.tools.hr_tools import (
        get_employee_details,
        get_attrition_risk,
        get_department_stats,
    )

    print(f"\n{'='*60}")
    print(f"INVESTIGATION REPORT — Employee {employee_id}")
    print("=" * 60)

    # Step 1: Get employee details
    print("\n[STEP 1] Getting employee details...")
    details_raw = get_employee_details(employee_id)
    details = json.loads(details_raw)

    if details["status"] == "not_found":
        print(f"  ERROR: {details['error']}")
        return {"status": "not_found"}

    print(f"  Department: {details['department']}")
    print(f"  Role:       {details['job_role']}")
    print(f"  Age:        {details['age']}")
    print(f"  Salary:     ${details['monthly_income']:,.0f}/month")
    print(f"  Overtime:   {details['overtime']}")
    print(f"  Tenure:     {details['years_at_company']} years")
    print(f"  Satisfaction: {details['satisfaction_score']:.1f}/4.0")
    print(f"  Status:     {details['attrition']}")

    # Step 2: Calculate attrition risk
    print("\n[STEP 2] Calculating attrition risk...")
    risk_raw = get_attrition_risk(employee_id)
    risk = json.loads(risk_raw)

    print(f"  Risk Level: {risk['risk_level']}")
    print(f"  Risk Score: {risk['risk_score']}/10")

    if risk["risk_factors"]:
        print(f"  Risk Factors:")
        for factor in risk["risk_factors"]:
            print(f"    → {factor}")
    else:
        print(f"  No major risk factors identified")

    print(f"  Recommendation: {risk['recommendation']}")

    # Step 3: Department benchmarks
    print("\n[STEP 3] Comparing to department peers...")
    dept_raw = get_department_stats(details["department"])
    dept = json.loads(dept_raw)

    salary_diff = details["monthly_income"] - dept["median_salary"]
    print(f"  Department:         {details['department']}")
    print(f"  Dept size:          {dept['employee_count']} employees")
    print(f"  Employee salary:    ${details['monthly_income']:,.0f}")
    print(f"  Dept median salary: ${dept['median_salary']:,.0f}")
    print(f"  Vs median:          ${salary_diff:+,.0f}")
    print(f"  Dept attrition:     {dept['attrition_rate']:.1%}")
    print(f"  Dept overtime rate: {dept['overtime_rate']:.1%}")

    # Step 4: Generate summary
    print(f"\n{'='*60}")
    print("INVESTIGATION SUMMARY")
    print("=" * 60)

    summary_lines = [
        f"Employee {employee_id} — {details['job_role']} "
        f"in {details['department']}",
        f"Attrition Risk: {risk['risk_level']} "
        f"(score: {risk['risk_score']})",
        f"Salary: ${details['monthly_income']:,.0f}/month "
        f"(${salary_diff:+,.0f} vs dept median)",
    ]

    if risk["risk_factors"]:
        summary_lines.append("Key risk factors:")
        for factor in risk["risk_factors"]:
            summary_lines.append(f"  - {factor}")

    summary_lines.append(
        f"\nRecommended Action: {risk['recommendation']}"
    )

    summary = "\n".join(summary_lines)
    print(summary)

    return {
        "status": "complete",
        "employee_id": employee_id,
        "details": details,
        "risk": risk,
        "department": dept,
        "summary": summary,
    }


def investigate_multiple(employee_ids: list[int]) -> list[dict]:
    """
    Investigate multiple employees.
    Shows batch investigation capability.

    Args:
        employee_ids: List of employee IDs

    Returns:
        List of investigation results
    """
    results = []
    for emp_id in employee_ids:
        result = investigate_employee(emp_id)
        results.append(result)
    return results


def get_high_risk_employees(top_n: int = 5) -> list[dict]:
    """
    Find the highest risk employees in the dataset.
    Scans a sample and returns top N by risk score.

    Args:
        top_n: Number of high risk employees to return

    Returns:
        List of high risk employee profiles
    """
    from src.agents.tools.hr_tools import (
        get_employee_details,
        get_attrition_risk,
    )

    print(f"\nScanning for top {top_n} highest risk employees...")

    scored = []

    # Scan first 50 employees as a sample
    for emp_id in range(1, 51):
        risk_raw = get_attrition_risk(emp_id)
        risk = json.loads(risk_raw)

        if risk["status"] == "success":
            scored.append({
                "employee_id": emp_id,
                "risk_level": risk["risk_level"],
                "risk_score": risk["risk_score"],
                "risk_factors": risk["risk_factors"],
            })

    # Sort by risk score
    scored.sort(key=lambda x: x["risk_score"], reverse=True)
    top = scored[:top_n]

    print(f"\nTop {top_n} Highest Risk Employees:")
    print("-" * 50)
    for emp in top:
        print(f"  Employee {emp['employee_id']:4d}: "
              f"{emp['risk_level']:6s} (score: {emp['risk_score']})")

    return top


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test 1: Investigate single employee
    result = investigate_employee(7)

    print(f"\n\n{'='*60}")
    print("SCANNING FOR HIGH RISK EMPLOYEES")
    print("=" * 60)

    # Test 2: Find high risk employees
    high_risk = get_high_risk_employees(top_n=5)

    print(f"\n{'='*60}")
    print("AGENT INVESTIGATION COMPLETE")
    print("=" * 60)
    print(f"Single investigation: Employee 7 — {result['risk']['risk_level']} risk")
    print(f"High risk scan: {len(high_risk)} employees flagged")