# src/agents/tools/hr_tools.py
"""
Tools for the HR investigation agent.
Each function is a tool the agent can call.
"""

import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Load dataset once at module level
_HR_DATA = None


def _get_hr_data() -> pd.DataFrame:
    """Load HR dataset (cached after first load)."""
    global _HR_DATA
    if _HR_DATA is None:
        from src.data.ingest import load_hr_data
        from src.data.clean import clean_hr_data
        _HR_DATA = clean_hr_data(load_hr_data())
    return _HR_DATA


def get_employee_details(employee_id: int) -> str:
    """
    Get details for a specific employee.
    Use this tool FIRST to get employee context.

    Args:
        employee_id: Integer from 1 to 1470

    Returns:
        JSON string with employee profile
    """
    df = _get_hr_data()

    if employee_id < 1 or employee_id > len(df):
        return json.dumps({
            "status": "not_found",
            "error": f"Employee {employee_id} not found"
        })

    row = df.iloc[employee_id - 1]

    return json.dumps({
        "status": "found",
        "employee_id": employee_id,
        "age": int(row["Age"]),
        "department": str(row["Department"]),
        "job_role": str(row["JobRole"]),
        "monthly_income": float(row["MonthlyIncome"]),
        "years_at_company": int(row["YearsAtCompany"]),
        "total_working_years": int(row["TotalWorkingYears"]),
        "job_satisfaction": int(row["JobSatisfaction"]),
        "overtime": "Yes" if row["OverTime"] == 1 else "No",
        "distance_from_home": int(row["DistanceFromHome"]),
        "attrition": "Left" if row["Attrition"] == 1 else "Active",
        "tenure_ratio": float(row["TenureRatio"]),
        "satisfaction_score": float(row["SatisfactionScore"]),
    }, indent=2)


def get_attrition_risk(employee_id: int) -> str:
    """
    Calculate attrition risk for an employee.
    Use this tool when asked about flight risk or retention.

    Args:
        employee_id: Integer from 1 to 1470

    Returns:
        JSON string with risk level and key factors
    """
    df = _get_hr_data()

    if employee_id < 1 or employee_id > len(df):
        return json.dumps({"status": "not_found"})

    row = df.iloc[employee_id - 1]

    # Calculate risk factors
    risk_factors = []
    risk_score = 0

    if row["OverTime"] == 1:
        risk_factors.append("Working overtime — 2.9x higher attrition risk")
        risk_score += 3

    if row["JobSatisfaction"] <= 2:
        risk_factors.append(f"Low job satisfaction: {row['JobSatisfaction']}/4")
        risk_score += 2

    if row["YearsAtCompany"] < 3:
        risk_factors.append(f"Short tenure: {row['YearsAtCompany']} years")
        risk_score += 2

    if row["DistanceFromHome"] > 20:
        risk_factors.append(f"Long commute: {row['DistanceFromHome']}km")
        risk_score += 1

    if row["MonthlyIncome"] < 3000:
        risk_factors.append(f"Below median salary: ${row['MonthlyIncome']:,.0f}")
        risk_score += 2

    # Risk level
    if risk_score >= 5:
        risk_level = "HIGH"
        recommendation = "Immediate retention conversation recommended"
    elif risk_score >= 3:
        risk_level = "MEDIUM"
        recommendation = "Monitor closely, discuss in next 1:1"
    else:
        risk_level = "LOW"
        recommendation = "No immediate action required"

    return json.dumps({
        "status": "success",
        "employee_id": employee_id,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
    }, indent=2)


def get_department_stats(department: str) -> str:
    """
    Get statistics for a specific department.
    Use this to compare an employee to their peers.

    Args:
        department: One of Sales, Research & Development,
                    Human Resources

    Returns:
        JSON string with department benchmarks
    """
    df = _get_hr_data()
    dept_df = df[df["Department"] == department]

    if len(dept_df) == 0:
        return json.dumps({
            "status": "not_found",
            "valid_departments": df["Department"].unique().tolist()
        })

    return json.dumps({
        "status": "found",
        "department": department,
        "employee_count": len(dept_df),
        "median_salary": float(dept_df["MonthlyIncome"].median()),
        "avg_satisfaction": float(dept_df["JobSatisfaction"].mean()),
        "attrition_rate": float(dept_df["Attrition"].mean()),
        "overtime_rate": float(dept_df["OverTime"].mean()),
        "avg_tenure": float(dept_df["YearsAtCompany"].mean()),
    }, indent=2)


def search_hr_policy(query: str) -> str:
    """
    Search HR policy documents for relevant information.
    Use this when asked about leave, overtime, or HR rules.

    Args:
        query: Policy question to search for

    Returns:
        Most relevant policy text found
    """
    from src.rag.document_processor import process_policy_documents
    from src.rag.chain import retrieve_relevant_chunks
    from openai import AzureOpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version="2024-02-01",
    )

    chunks = process_policy_documents()
    relevant = retrieve_relevant_chunks(query, chunks, client, top_k=2)

    results = [
        {
            "source": c["source"],
            "content": c["content"][:300],
            "relevance": round(c["score"], 3),
        }
        for c in relevant
    ]

    return json.dumps({
        "status": "found",
        "query": query,
        "results": results,
    }, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing HR Tools...")
    print("\n--- get_employee_details(1) ---")
    print(get_employee_details(1))

    print("\n--- get_attrition_risk(1) ---")
    print(get_attrition_risk(1))

    print("\n--- get_department_stats('Sales') ---")
    print(get_department_stats("Sales"))