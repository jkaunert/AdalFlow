# High-Level API Design: Intuitive Program & Agent Evolution in AdalFlow

**Date:** 2026-05-25  
**POC:** AdaL & Engineering Team  
**Status:** Draft / Proposal  

---

## 1. Design Goal
To provide developers with a simple, high-level, and intuitive interface to define LLM components, agents, and pipelines, and automatically optimize/evolve them (using text gradients, few-shot generation, or genetic algorithms/GEPA) with minimal script-level scaffolding—similar to `openevolve` but deeply integrated with AdalFlow's performance-oriented engine.

---

## 2. API Showcase

Here is how a developer would use this new abstraction to define a multi-agent system, artifacts, evaluators, and run an evolutionary loop in a single, clean script:

```python
import adalflow as adal
from adalflow.optim.evolution import run_evolution, evolve_function

# 1. Define Agent Profiles and Artifact relations as modular units
@adal.agent(name="planner", model="gpt-4o")
class PlannerAgent:
    """You are a precise task planner. Break the request into 3 clear steps."""
    def __call__(self, task: str) -> str:
        return self.generate(task)

@adal.agent(name="coder", model="claude-3-5-sonnet")
class CodingAgent:
    """You are an advanced Python coding agent. Implement clean, tested functions."""
    def __call__(self, plan: str) -> str:
        return self.generate(plan)

# 2. Wire them up in an intuitive pipeline DAG (ADAG)
class CodingPipeline(adal.Pipeline):
    def __init__(self):
        super().__init__()
        self.planner = PlannerAgent()
        self.coder = CodingAgent()

    def call(self, user_query: str) -> str:
        # Define flow and sequential artifact passing simply
        plan = self.planner(user_query)
        code = self.coder(plan)
        return code

# 3. Instantiate the pipeline
pipeline = CodingPipeline()

# 4. Define your evaluator (can be unit tests or programmatic metrics)
def evaluate_pipeline(pipeline_instance) -> float:
    # Run the pipeline on a small test benchmark
    sample_task = "Write a fast fibonacci function"
    generated_code = pipeline_instance(sample_task)
    
    # Run tests on generated_code
    try:
        exec(generated_code)
        # Test assertion check
        assert fib(5) == 5
        return 1.0
    except Exception:
        return 0.0

# 5. One-line Evolution run!
# AdalFlow automatically discovers all prompt parameters in PlannerAgent and CodingAgent,
# mutates/crosses-over their profiles, runs evaluations, and converges on the optimal prompts.
best_pipeline = run_evolution(
    pipeline,
    evaluator=evaluate_pipeline,
    iterations=20,
    strategy="gepa" # or "text-grad"
)

# Print out the evolved optimal system prompts
print("Optimized Planner Profile:", best_pipeline.planner.profile)
print("Optimized Coder Profile:", best_pipeline.coder.profile)
```

---

## 3. Underlying Mechanics

### 3.1 Class Decorators & Dynamic Parameters
The `@adal.agent` decorator dynamically converts the docstring and system instructions into an AdalFlow `Parameter(requires_grad=True)`. This keeps the definition extremely elegant and visually clean while fully preserving the PyTorch-style computational graph underneath.

```python
def agent(name: str, model: str):
    def decorator(cls):
        # Dynamically inject AdalFlow Component properties
        class WrappedAgent(Component):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.profile = Parameter(
                    data=cls.__doc__,
                    requires_grad=True,
                    role_desc=f"Profile instructions for {name}",
                    param_type=ParameterType.PROMPT
                )
                self.generator = Generator(
                    model_client=get_default_client(),
                    model_kwargs={"model": model}
                )
            
            def generate(self, input_text: str) -> str:
                return self.generator(
                    prompt_kwargs={"system_prompt": self.profile.data, "input": input_text}
                ).data
                
        return WrappedAgent
    return decorator
```

### 3.2 High-Level `run_evolution` Entrypoint
The `run_evolution` function wraps AdalFlow's optimizer suite:
- Analyzes the provided `Pipeline` or `Component` to extract all parameters.
- Wraps the custom `evaluator` function as an AdalFlow metric tracker.
- Instantiates either `GEPA` (for evolutionary mutation/crossover of prompts/code) or `TGDOptimizer` (for text-gradient feedback iterations).
- Manages the mutation selection of parameters behind the scenes and returns the optimized component state.

---

## 4. Why This is Perfect for AdalFlow
- **Developer Delight:** Hides the boilerplate of dataset loaders, trainers, and optimizer parameter registration, making simple experiments take seconds to write.
- **Backward Compatible:** Since the decorated agents and pipelines inherit directly from `Component`, they remain 100% compatible with existing lower-level APIs (like MLflow tracing and custom trainers).
