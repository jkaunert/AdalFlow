"""
AdalFlow Implementation of an Autonomous Research Agent (AutoResearchClaw-style).
This demo showcases how AdalFlow's Component architecture, Prompt Parameters, and Custom Evaluators 
can model a multi-stage autonomous research pipeline with Human-in-the-Loop (HITL) gates, 
Self-Healing code sandboxes, and run-to-run Optimization.
"""

import sys
import time
from typing import Dict, Any, List, Optional
from adalflow.core.component import Component
from adalflow.optim.parameter import Parameter, ParameterType

# =====================================================================
# 1. Pipeline Building Blocks: AdalFlow Stages and Components
# =====================================================================

class LiteratureAgent(Component):
    """
    Search and analyze relevant literature using scholarly APIs.
    """
    def __init__(self):
        super().__init__()
        self.system_prompt = Parameter(
            data="You are an expert literature surveyor. Identify key research papers, extract findings, and isolate open research gaps.",
            requires_opt=True,
            role_desc="Instructions for LiteratureAgent to maximize relevance and coverage of references.",
            param_type=ParameterType.PROMPT
        )

    def call(self, topic: str) -> List[Dict[str, str]]:
        print(f"\n[LiteratureAgent] Searching real & virtual databases for topic: '{topic}'")
        # In a real environment, we would call OpenAlex or Semantic Scholar APIs here.
        # Returning mock reference results.
        return [
            {"title": "AdalFlow: Auto-optimizing LLM pipelines", "authors": "Yang et al.", "year": "2024"},
            {"title": "AutoResearchClaw: Autonomous Research Agents", "authors": "Liu et al.", "year": "2026"}
        ]


class HypothesisAgent(Component):
    """
    Synthesize literature findings and design testable hypotheses.
    """
    def __init__(self):
        super().__init__()
        self.system_prompt = Parameter(
            data="Formulate highly novel, testable machine learning hypotheses. Focus on parameter-efficient tuning.",
            requires_opt=True,
            role_desc="Synthesizing instructions to generate high-novelty hypotheses.",
            param_type=ParameterType.PROMPT
        )

    def call(self, literature_findings: List[Dict[str, str]]) -> str:
        print("[HypothesisAgent] Analyzing findings to generate a novel research hypothesis...")
        return "Hypothesis: AdalFlow's Textual Gradient Descent (TGD) is more sample-efficient than standard RLHF on reasoning tasks."


class SecureSandboxExecutor(Component):
    """
    Secure executor which runs the generated code in a sandbox, catches errors, 
    and returns traceback feedback for self-healing.
    """
    def call(self, code: str) -> Dict[str, Any]:
        print("[Sandbox] Executing candidate experiment code in isolated sandbox...")
        # Simulated run with error tracing
        if "eval_accuracy" in code:
            return {"status": "success", "metrics": {"eval_accuracy": 0.885}, "logs": "All checks passed."}
        else:
            return {"status": "failed", "traceback": "NameError: name 'eval_accuracy' is not defined", "logs": "Runtime execution failed."}


class SelfHealingCoder(Component):
    """
    Self-healing execution module that writes code and iteratively repairs tracebacks.
    """
    def __init__(self):
        super().__init__()
        self.system_prompt = Parameter(
            data="Write clean, modular, runnable training scripts. Ensure metrics like 'eval_accuracy' are tracked.",
            requires_opt=True,
            role_desc="System guidelines for compiling bug-free code.",
            param_type=ParameterType.PROMPT
        )
        self.sandbox = SecureSandboxExecutor()

    def call(self, hypothesis: str) -> Dict[str, Any]:
        print("[Self-Healing Coder] Generating initial code based on hypothesis...")
        code = "import torch\n# Running standard training sequence\nval_loss = 0.1"
        
        # Iteration 1: Sandbox run fails (simulated)
        result = self.sandbox(code)
        if result["status"] == "failed":
            print(f"[Self-Healing Coder] Caught Error: {result['traceback']}. Initiating LLM self-healing repair loop...")
            # Repairing code
            code += "\neval_accuracy = 0.89"
            result = self.sandbox(code)
            
        print(f"[Self-Healing Coder] Code execution finalized. Status: {result['status']}")
        return result


class HumanInTheLoopGate:
    """
    A lightweight HITL gate to pause execution, solicit user feedback, 
    or auto-approve in head-less mode.
    """
    @staticmethod
    def approve(stage_name: str, candidate_output: Any, mode: str = "co-pilot") -> Any:
        print(f"\n--- [HITL Gate: {stage_name}] Mode: {mode} ---")
        print(f"Candidate Proposal:\n{candidate_output}")
        if mode == "auto":
            print("Auto-approved.")
            return candidate_output
        
        # Emulating standard CLI decision prompt
        print("[HITL] Options: [a] Approve, [e] Edit, [c] Collaborate")
        print("[HITL] Decision: Approved by default in this demo.")
        return candidate_output


# =====================================================================
# 2. Main Orchestrator DAG Component
# =====================================================================

class AutoResearchPipeline(Component):
    """
    23-stage equivalent pipeline defined natively as an AdalFlow Component.
    """
    def __init__(self, hitl_mode: str = "co-pilot"):
        super().__init__()
        self.lit_agent = LiteratureAgent()
        self.hyp_agent = HypothesisAgent()
        self.coder = SelfHealingCoder()
        self.hitl_mode = hitl_mode

    def call(self, topic: str) -> Dict[str, Any]:
        print(f"\n===== Initiating AutoResearchPipeline for Topic: '{topic}' =====")
        
        # Stage 1-6: Literature collection
        papers = self.lit_agent(topic)
        
        # Stage 7-8: Hypothesis Generation & Multi-agent debate
        raw_hypothesis = self.hyp_agent(papers)
        
        # Stage 9: Human-In-The-Loop gate
        hypothesis = HumanInTheLoopGate.approve("HYPOTHESIS_GEN", raw_hypothesis, self.hitl_mode)
        
        # Stage 10-13: Code Gen & Sandbox Execution + Self-Healing
        exp_result = self.coder(hypothesis)
        
        # Stage 14-23: Final output compilation
        paper_status = "Successfully drafted NeurIPS-compatible LaTeX paper."
        
        return {
            "hypothesis": hypothesis,
            "experiment_metrics": exp_result.get("metrics", {}),
            "output_status": paper_status
        }


# =====================================================================
# 3. Demonstration & Mock Benchmark Execution
# =====================================================================

def demo_run():
    # 1. Instantiate the modular pipeline
    pipeline = AutoResearchPipeline(hitl_mode="co-pilot")
    
    # 2. Execute on a target topic
    topic = "Optimize AdalFlow Prompting dynamically via self-referencing feedback loops."
    result = pipeline(topic)
    
    print("\n" + "="*50)
    print("DEMO EXECUTION SUMMARY:")
    print(f"Hypothesis: {result['hypothesis']}")
    print(f"Metrics:    {result['experiment_metrics']}")
    print(f"Status:     {result['output_status']}")
    print("="*50 + "\n")

    # 3. Print out the Comparative Benchmark analysis
    print_benchmark_report()

def print_benchmark_report():
    print("""
=========================================================================================
            ARC-BENCH SMOKE BENCHMARK STATUS
=========================================================================================
This demo is an AdalFlow-style autonomous research pipeline prototype. 

To collect reproducible smoke benchmark metrics, run:

python benchmarks/arc_bench/run_adalflow_smoke.py \
  --topics benchmarks/arc_bench/topics_smoke.jsonl \
  --output artifacts/arc_bench/adalflow_smoke

Then generate the report:

python benchmarks/arc_bench/report.py \
  --input artifacts/arc_bench/adalflow_smoke \
  --output artifacts/arc_bench/adalflow_smoke_report.md
=========================================================================================
""")

if __name__ == "__main__":
    demo_run()
