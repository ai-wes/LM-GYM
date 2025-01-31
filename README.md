# LLM GYM 🧠

A comprehensive framework for training and evaluating Large Language Model cognitive strategies through reinforcement learning environments. LLM GYM provides a collection of specialized environments designed to elicit and measure different aspects of LLM reasoning and capabilities.

## Features 🌟

### Core Environments
- **Meta-Reasoning Environment** (`MetaReasoningEnv`): Train LLMs in explicit reasoning and reflection
  - Chain-of-thought generation
  - Assumption verification
  - Bias detection
  - Uncertainty quantification

- **Logic Puzzle Environment** (`LogicPuzzleEnv`): Develop structured logical reasoning
  - Premise-based deduction
  - Variable relationship understanding
  - Multi-step logical inference
  - Query-based exploration

- **Dialogue Environment** (`DialogueEnv`): Practice conversation and social interaction
  - Multiple conversation scenarios (Q&A, negotiation, debate)
  - Required conversation elements (politeness, coherence)
  - Emotional awareness training
  - Cultural sensitivity development

- **Memory Buffer Environment** (`MemoryBufferEnv`): Train continual learning and memory
  - Persistent memory across episodes
  - Information storage and retrieval
  - Topic-based organization
  - Temporal and categorical memory

### Safety and Ethics
- Comprehensive safety policy framework
- Content safety monitoring
- Fairness and bias detection
- Privacy protection
- Real-time safety checks during training

### Enhanced Curriculum Learning
- Dynamic difficulty adjustment
- Performance-based task selection
- Adaptive parameter scaling
- Progress tracking and analysis
- Multi-dimensional task templates

### Advanced Monitoring
- Real-time performance visualization
- Comprehensive metrics logging
- Learning curve analysis
- Task distribution insights
- Safety violation tracking
- Automated report generation

### Adaptive Rewards
- Multi-objective reward computation
- Dynamic weight adjustment
- Component-based reward design
- Performance-based adaptation
- Novelty and efficiency rewards

### Distributed Training
- Parallel environment execution
- Scalable worker management
- Synchronized metric collection
- Distributed safety checks
- Resource-efficient training

## Installation 🚀

```bash
# Clone the repository
git clone https://github.com/ai-wes/llm-gym.git
cd llm-gym

python -m venv llm_gym_env
source llm_gym_env/bin/activate
```
    
First, upgrade pip and required tools:

```bash
python -m pip install --upgrade setuptools pip wheel
```
### Then Install Pytorch with Cuda

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Install the NVIDIA package index

```bash
python -m pip install nvidia-pyindex
```


```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start 🎯

### Basic Usage
```python
from llm_gym.envs import MetaReasoningEnv
from llm_gym.evaluation import EvaluationPipeline

# Create environment
env = MetaReasoningEnv(use_curriculum=True)

# Create evaluation pipeline
pipeline = EvaluationPipeline(env)

# Run evaluation
results = pipeline.evaluate(num_episodes=10)
```

### Multi-Environment Demo
```python
from llm_gym.envs import (
    MetaReasoningEnv,
    LogicPuzzleEnv,
    DialogueEnv,
    MemoryBufferEnv
)

# Create environments
meta_env = MetaReasoningEnv()
logic_env = LogicPuzzleEnv()
dialogue_env = DialogueEnv()
memory_env = MemoryBufferEnv()

# Use environments for different cognitive tasks
```

## Environment Details 📚

### Meta-Reasoning Environment
```python
env = MetaReasoningEnv(
    use_curriculum=True,
    use_human_feedback=False,
    max_steps=10
)

# Available actions:
# - THINK <reasoning>: Provide step-by-step reasoning
# - ANSWER <solution>: Submit final answer
```

### Logic Puzzle Environment
```python
env = LogicPuzzleEnv(
    use_curriculum=True,
    max_steps=10
)

# Available actions:
# - SOLVE <solution>: Attempt to solve the puzzle
# - QUERY <question>: Ask about relationships
```

### Dialogue Environment
```python
env = DialogueEnv(
    use_curriculum=True,
    use_human_feedback=True,
    max_turns=10
)

# Features:
# - Multiple conversation scenarios
# - Required elements tracking
# - Human feedback integration
```

### Memory Buffer Environment
```python
env = MemoryBufferEnv(
    use_curriculum=True,
    max_steps=10,
    memory_size=100
)

# Available actions:
# - STORE <content>: Store information
# - RETRIEVE <query>: Search memory
# - ANSWER <response>: Respond to query
```

## Curriculum Generation 📈

### Custom Task Templates
```yaml
# tasks.yaml
logic_puzzle:
  templates:
    - "If A then B"
    - "Either A or B"
  variables:
    - ["Raining", "Wet"]
    - ["Studying", "Pass"]

dialogue:
  scenarios:
    - type: "negotiation"
      context: "Salary discussion"
```

```python
generator = CurriculumGenerator(
    initial_difficulty=0.1,
    task_templates_path="tasks.yaml"
)
```

## Evaluation Pipeline 📊

### Basic Evaluation
```python
pipeline = EvaluationPipeline(env, log_dir="logs")
results = pipeline.evaluate(num_episodes=100)

# Plot learning curve
pipeline.plot_learning_curve("learning_curve.png")
```

### Metrics Tracked
- Episode rewards
- Task completion rates
- Chain-of-thought length
- Success rate progression
- Difficulty progression

## Reward System 🎯

### Custom Reward Components
```python
from llm_gym.utils.rewards import RewardComponent

class CustomReward(RewardComponent):
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="custom",
            weight=weight,
            compute_fn=self._compute_reward
        )
    
    def _compute_reward(self, state: Dict[str, Any]) -> float:
        # Custom reward logic
        return reward_value
```

## Project Structure 📁
```
llm_gym/
├── envs/                    # Environment implementations
│   ├── base.py             # Base environment class
│   ├── meta_reasoning.py   # Meta-reasoning environment
│   ├── logic_puzzle.py     # Logic puzzle environment
│   ├── dialogue.py         # Dialogue environment
│   └── memory.py           # Memory buffer environment
├── curriculum/             # Curriculum management
│   ├── generator.py        # Task generation
│   └── difficulty.py       # Difficulty adjustment
├── evaluation/             # Evaluation tools
│   ├── pipeline.py         # Evaluation pipeline
│   └── metrics.py          # Evaluation metrics
├── safety/                 # Safety and ethics
│   └── __init__.py
├── monitoring/             # Monitoring tools
│   └── visualization.py    # Performance visualization
└── utils/                  # Utility functions
    ├── logging.py          # Logging utilities
    └── rewards.py          # Reward management
```

## Contributing 🤝

Contributions are welcome! Here are some ways you can contribute:
- Add new environments
- Improve task generation
- Enhance reward mechanisms
- Add evaluation metrics
- Improve documentation

## License 📄

MIT License - see [LICENSE](LICENSE) for details.

## Citation 📚

If you use LLM GYM in your research, please cite:

```bibtex
@software{llm_gym2024,
  title = {LLM GYM: A Framework for Training LLM Cognitive Strategies},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llm-gym}
}
``` 



