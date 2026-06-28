# src/agents/audit_graph.py
"""
LangGraph Multi-Agent Payroll Audit System.

Architecture:
  Supervisor Agent
    ├── Employee Agent   (fetches employee details)
    ├── Risk Agent       (calculates attrition risk)
    └── Report Agent     (generates final report)

Patterns demonstrated:
  - LangGraph StateGraph
  - Shared TypedDict state
  - Conditional routing
  - HITL interrupt pattern
  - Parallel-ready node design
"""

import json
import logging
from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


# ─── State Definition ─────────────────────────────────────────
class AuditState(TypedDict):
    """
    Shared state passed between all agents.
    Every agent reads from and writes to this state.

    Annotated[list, operator.add] means:
    When multiple agents write to this field,
    their outputs are COMBINED (not overwritten).
    This enables parallel execution safely.
    """
    employee_id: int
    employee_details: dict
    risk_assessment: dict
    department_stats: dict
    final_report: str
    requires_human_review: bool
    messages: Annotated[list[str], operator.add]
    status: str


# ─── Agent Nodes ──────────────────────────────────────────────
def supervisor_node(state: AuditState) -> dict:
    """
    Supervisor: validates input and initiates audit.
    First node in the graph.
    """
    employee_id = state["employee_id"]
    logger.info(f"[SUPERVISOR] Starting audit for employee {employee_id}")

    return {
        "messages": [f"Supervisor: Starting audit for employee {employee_id}"],
        "status": "in_progress",
    }


def employee_agent_node(state: AuditState) -> dict:
    """
    Employee Agent: fetches full employee profile.
    Runs the get_employee_details tool.
    """
    from src.agents.tools.hr_tools import get_employee_details

    employee_id = state["employee_id"]
    logger.info(f"[EMPLOYEE AGENT] Fetching details for {employee_id}")

    details_raw = get_employee_details(employee_id)
    details = json.loads(details_raw)

    message = (
        f"Employee Agent: Retrieved profile — "
        f"{details.get('job_role', 'Unknown')} in "
        f"{details.get('department', 'Unknown')}"
    )

    return {
        "employee_details": details,
        "messages": [message],
    }


def risk_agent_node(state: AuditState) -> dict:
    """
    Risk Agent: calculates attrition risk.
    Runs the get_attrition_risk tool.
    """
    from src.agents.tools.hr_tools import get_attrition_risk

    employee_id = state["employee_id"]
    logger.info(f"[RISK AGENT] Calculating risk for {employee_id}")

    risk_raw = get_attrition_risk(employee_id)
    risk = json.loads(risk_raw)

    message = (
        f"Risk Agent: Risk level = {risk.get('risk_level')} "
        f"(score: {risk.get('risk_score')})"
    )

    # Flag for human review if HIGH risk
    requires_review = risk.get("risk_level") == "HIGH"

    return {
        "risk_assessment": risk,
        "requires_human_review": requires_review,
        "messages": [message],
    }


def department_agent_node(state: AuditState) -> dict:
    """
    Department Agent: fetches department benchmarks.
    Compares employee to their peers.
    """
    from src.agents.tools.hr_tools import get_department_stats

    details = state.get("employee_details", {})
    department = details.get("department", "Sales")

    logger.info(f"[DEPT AGENT] Fetching stats for {department}")

    dept_raw = get_department_stats(department)
    dept = json.loads(dept_raw)

    salary = details.get("monthly_income", 0)
    median = dept.get("median_salary", 0)
    diff = salary - median

    message = (
        f"Department Agent: Employee salary ${salary:,.0f} "
        f"vs dept median ${median:,.0f} "
        f"(${diff:+,.0f})"
    )

    return {
        "department_stats": dept,
        "messages": [message],
    }


def hitl_node(state: AuditState) -> dict:
    """
    HITL (Human In The Loop) node.
    Pauses for human review when risk is HIGH.

    In production: this sends a notification and waits.
    In this demo: simulates the approval decision.
    """
    logger.info("[HITL] Human review required — pausing for approval")

    risk = state.get("risk_assessment", {})
    details = state.get("employee_details", {})

    print(f"\n{'='*60}")
    print("⚠️  HUMAN REVIEW REQUIRED")
    print("=" * 60)
    print(f"Employee ID:  {state['employee_id']}")
    print(f"Risk Level:   {risk.get('risk_level')}")
    print(f"Department:   {details.get('department')}")
    print(f"Risk Factors:")
    for factor in risk.get("risk_factors", []):
        print(f"  → {factor}")
    print(f"\nIn production: notification sent to HR manager.")
    print(f"Simulating approval for demo purposes...")
    print("=" * 60)

    return {
        "messages": ["HITL: Human review completed — approved for report"],
        "status": "approved",
    }


def report_agent_node(state: AuditState) -> dict:
    """
    Report Agent: generates the final audit report.
    Combines all information into a structured report.
    """
    logger.info("[REPORT AGENT] Generating final report")

    employee_id = state["employee_id"]
    details = state.get("employee_details", {})
    risk = state.get("risk_assessment", {})
    dept = state.get("department_stats", {})

    salary = details.get("monthly_income", 0)
    median = dept.get("median_salary", 0)
    diff = salary - median

    report = f"""
SMARTPAYROLL AI — AUDIT REPORT
{'='*50}
Employee ID:    {employee_id}
Role:           {details.get('job_role', 'Unknown')}
Department:     {details.get('department', 'Unknown')}
Age:            {details.get('age', 'Unknown')}
Status:         {details.get('attrition', 'Unknown')}

FINANCIAL ANALYSIS:
  Monthly Salary:     ${salary:,.0f}
  Department Median:  ${median:,.0f}
  Variance:           ${diff:+,.0f}

ATTRITION RISK:
  Risk Level:   {risk.get('risk_level', 'Unknown')}
  Risk Score:   {risk.get('risk_score', 0)}/10
  Key Factors:
{chr(10).join(f"    → {f}" for f in risk.get('risk_factors', []))}

RECOMMENDATION:
  {risk.get('recommendation', 'No recommendation')}

DEPARTMENT CONTEXT:
  Team Size:        {dept.get('employee_count', 0)} employees
  Dept Attrition:   {dept.get('attrition_rate', 0):.1%}
  Dept OT Rate:     {dept.get('overtime_rate', 0):.1%}

AUDIT TRAIL:
{chr(10).join(f"  {msg}" for msg in state.get('messages', []))}
{'='*50}
"""

    return {
        "final_report": report,
        "messages": ["Report Agent: Final report generated"],
        "status": "complete",
    }


# ─── Routing Logic ────────────────────────────────────────────
def route_after_risk(
    state: AuditState,
) -> Literal["hitl_node", "department_agent_node"]:
    """
    Conditional routing after risk assessment.

    HIGH risk → human review first (HITL)
    LOW/MEDIUM → proceed directly to department analysis
    """
    if state.get("requires_human_review", False):
        logger.info("Routing to HITL — HIGH risk detected")
        return "hitl_node"
    else:
        logger.info("Routing to department agent — risk acceptable")
        return "department_agent_node"


# ─── Build the Graph ──────────────────────────────────────────
def build_audit_graph() -> StateGraph:
    """
    Build and compile the LangGraph audit workflow.

    Flow:
    START
      → supervisor_node
      → employee_agent_node
      → risk_agent_node
      → [conditional] hitl_node OR department_agent_node
      → department_agent_node
      → report_agent_node
    END
    """
    builder = StateGraph(AuditState)

    # Add nodes
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("employee_agent_node", employee_agent_node)
    builder.add_node("risk_agent_node", risk_agent_node)
    builder.add_node("hitl_node", hitl_node)
    builder.add_node("department_agent_node", department_agent_node)
    builder.add_node("report_agent_node", report_agent_node)

    # Add edges (the flow)
    builder.add_edge(START, "supervisor_node")
    builder.add_edge("supervisor_node", "employee_agent_node")
    builder.add_edge("employee_agent_node", "risk_agent_node")

    # Conditional routing after risk assessment
    builder.add_conditional_edges(
        "risk_agent_node",
        route_after_risk,
        {
            "hitl_node": "hitl_node",
            "department_agent_node": "department_agent_node",
        }
    )

    # After HITL → department analysis
    builder.add_edge("hitl_node", "department_agent_node")

    # Department → final report
    builder.add_edge("department_agent_node", "report_agent_node")
    builder.add_edge("report_agent_node", END)

    return builder.compile()


def run_audit(employee_id: int) -> dict:
    """
    Run the complete audit workflow for one employee.

    Args:
        employee_id: Employee ID to audit

    Returns:
        Final state with complete audit report
    """
    graph = build_audit_graph()

    initial_state = {
        "employee_id": employee_id,
        "employee_details": {},
        "risk_assessment": {},
        "department_stats": {},
        "final_report": "",
        "requires_human_review": False,
        "messages": [],
        "status": "pending",
    }

    logger.info(f"Starting LangGraph audit for employee {employee_id}")
    final_state = graph.invoke(initial_state)

    return final_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("LANGGRAPH MULTI-AGENT AUDIT SYSTEM")
    print("=" * 60)

    # Test with HIGH risk employee (employee 7)
    print("\nTest 1: HIGH risk employee (7)")
    result = run_audit(7)
    print(result["final_report"])

    # Test with different employee
    print("\nTest 2: Different employee (1)")
    result2 = run_audit(1)
    print(result2["final_report"])

    print("\n" + "=" * 60)
    print("LANGGRAPH AUDIT COMPLETE")
    print("=" * 60)
    print(f"Test 1 status: {result['status']}")
    print(f"Test 2 status: {result2['status']}")