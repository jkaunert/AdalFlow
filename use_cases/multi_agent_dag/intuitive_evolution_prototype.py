"""
Prototype: Intuitive Agent and Program Evolution API on top of AdalFlow.
This prototype showcases the decorator @agent, the Pipeline class, and run_evolution
using a mock/simplified evolutionary mutator.
"""

from typing import Dict, Any, Callable
from adalflow.core.component import Component
from adalflow.optim.parameter import Parameter, ParameterType

# 1. High-Level Abstractions

def agent(name: str, model: str):
    """
    A decorator that transforms a user-defined class into an AdalFlow Component,
    converting its docstring into an optimizable Parameter.
    """
    def decorator(cls):
        class WrappedAgent(Component):
            def __init__(self, *args, **kwargs):
                super().__init__()
                # Convert the docstring of the class into an optimizable Prompt Parameter
                self.profile = Parameter(
                    data=cls.__doc__.strip() if cls.__doc__ else "You are a helpful assistant.",
                    requires_opt=True,
                    role_desc=f"Profile instructions for agent: {name}",
                    param_type=ParameterType.PROMPT
                )
                self.name = name
                self.model = model
                # Instantiating original class to bind methods
                self._user_inst = cls(*args, **kwargs)

            def generate(self, input_text: str) -> str:
                # Simulating dynamic LLM generation using the current system profile state
                print(f"[{self.name} ({self.model})] Running generation with profile: '{self.profile.data}'")
                # In real life, this hooks into: return self.generator(...)
                return f"Result of processing '{input_text}' using persona '{self.profile.data}'"

            def __call__(self, *args, **kwargs) -> Any:
                # Direct calls to __call__ on the instance are forwarded
                # We dynamically bind the generate helper so user functions can use it easily
                self._user_inst.generate = self.generate
                return self._user_inst.__call__(*args, **kwargs)

        return WrappedAgent
    return decorator


class Pipeline(Component):
    """Base class for wiring up pipelines."""
    def __init__(self):
        super().__init__()


def run_evolution(
    pipeline: Component,
    evaluator: Callable[[Component], float],
    iterations: int = 5,
    strategy: str = "gepa"
) -> Component:
    """
    A simplified evolutionary mutator running directly on the AdalFlow Component.
    In production, this translates to the full GEPA / TGDOptimizer engine.
    """
    print(f"\n--- Starting Evolution Loop (Strategy: {strategy}, Iterations: {iterations}) ---")
    
    # Locate all trainable parameters in the pipeline
    trainable_params = [p for p in pipeline.parameters() if p.requires_opt]
    print(f"Discovered {len(trainable_params)} trainable agent profiles:")
    for p in trainable_params:
        print(f" - {p.role_desc}: '{p.data}'")

    best_score = evaluator(pipeline)
    print(f"Initial Baseline Score: {best_score}")

    # Simulated evolution steps
    for step in range(1, iterations + 1):
        print(f"\n[Step {step}/{iterations}] Mutating profiles...")
        
        # 1. Mutate: In production, we send the prompt + current score + critique to LLM
        # For this prototype, we mock mutation by appending dynamic guidance hints
        original_states = []
        for param in trainable_params:
            original_states.append((param, param.data))
            param.data += f" (Optimized Guidance Step {step})"

        # 2. Evaluate candidate mutations
        score = evaluator(pipeline)
        print(f"Candidate Score: {score}")

        # 3. Selection
        if score >= best_score:
            best_score = score
            print(f"New Best State Found! Keeping mutations.")
        else:
            # Revert mutations
            print(f"Score dropped. Reverting mutations.")
            for param, orig_val in original_states:
                param.data = orig_val

    print(f"\n--- Evolution Complete. Best Final Score: {best_score} ---")
    return pipeline


# 2. Showcase Usage

if __name__ == "__main__":
    # Define Agents using standard Docstrings as system instructions
    @agent(name="planner", model="gpt-4o")
    class PlannerAgent:
        """You are a precise task planner. Break the request into 3 clear steps."""
        def __call__(self, task: str) -> str:
            # self.generate is dynamically supplied by the decorator
            return self.generate(task)

    @agent(name="coder", model="claude-3-5-sonnet")
    class CodingAgent:
        """You are an advanced Python coding agent. Implement clean, tested functions."""
        def __call__(self, plan: str) -> str:
            return self.generate(plan)

    # Wire them up as a simple pipeline DAG
    class CodingPipeline(Pipeline):
        def __init__(self):
            super().__init__()
            self.planner = PlannerAgent()
            self.coder = CodingAgent()

        def call(self, user_query: str) -> str:
            plan = self.planner(user_query)
            code = self.coder(plan)
            return code

    # Instantiate pipeline
    my_pipeline = CodingPipeline()

    # Define simple mock evaluator
    def mock_evaluator(pipeline_instance: Component) -> float:
        # Run the pipeline
        output = pipeline_instance("Write a fast fibonacci function")
        print(f"Pipeline output: {output}")
        
        # Scoring logic: prefer shorter, concise instructions
        total_len = sum(len(p.data) for p in pipeline_instance.parameters() if p.requires_opt)
        # Mock score that peaks when total length matches a target or increases with optimization
        return float(100.0 / total_len) if total_len > 0 else 0.0

    # Execute evolution
    evolved_pipeline = run_evolution(
        my_pipeline,
        evaluator=mock_evaluator,
        iterations=3,
        strategy="gepa"
    )

    print("\n=== Optimized Agent Profiles ===")
    print("Planner Profile:", evolved_pipeline.planner.profile.data)
    print("Coder Profile:", evolved_pipeline.coder.profile.data)
