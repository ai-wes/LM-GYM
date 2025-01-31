# Data Content and Quality Guidelines

**Example data constructs are located in the `llm_gym/data/` directory.**

## Logic Puzzle Environment:
High-quality logic puzzle examples should contain clear, unambiguous premises that form complete logical chains. Each example should include a mix of direct and indirect logical relationships, with the simpler puzzles using 2-3 clear premises and more complex ones incorporating 4-5 premises with transitive or contrapositive reasoning. Variables should be clearly defined with real-world context (like the TechCorp example) rather than abstract symbols. The solution path should break down the reasoning into discrete, verifiable steps that demonstrate the logical progression. Invalid conclusions should represent common logical fallacies or misapplications of the premises, serving as valuable teaching examples. Query templates should guide the problem-solving process from multiple angles.

## Dialogue Environment:

Strong dialogue examples should mirror realistic conversations with natural language patterns and authentic scenarios. Each example should include clear emotional dynamics (like the frustrated streaming customer) and specific business contexts (like the project deadline negotiation). Required elements should go beyond basic pleasantries to include sophisticated communication techniques such as active listening, empathy, and problem reframing. The conversation flow should demonstrate how to handle both standard progressions and challenging situations, with alternative paths covering common but difficult scenarios. Success criteria should be specific and measurable, while prohibited responses should reflect common communication pitfalls. The difficulty progression should be reflected in the complexity of the scenario, stakeholder needs, and emotional dynamics.

## Memory Buffer Environment:

Well-constructed memory buffer examples should feature interconnected information items that require both simple recall and complex relationship understanding. The information should be factually accurate and include clear relationships between items that can be tested through queries. Examples should vary in their buffer constraints to test different memory management strategies, from simple cases with few items to complex scenarios requiring strategic information prioritization. Queries should progress from basic fact checking to questions requiring multiple retrieval and synthesis steps. The evaluation metrics should reflect both accuracy and efficiency, with clear benchmarks for what constitutes good performance under different constraints.

## Meta-Reasoning Environment:

Exemplary meta-reasoning scenarios should present complex, real-world problems with multiple valid approaches and clear trade-offs. Each example should include rich context and meaningful constraints that shape the decision space. The reasoning components should demonstrate sophisticated analytical thinking, including explicit assumption identification, structured verification steps, and clear inference chains. Potential biases and uncertainty factors should be specific to the scenario rather than generic, with concrete mitigation strategies. Solution paths should demonstrate different valid approaches with clear justifications for each step and realistic confidence assessments. The evaluation criteria should holistically assess both the reasoning process and the practical viability of the solution, incorporating multiple dimensions of quality beyond just logical correctness.





### Implementation Notes:

1. **Data Collection Sources:**
   - Academic logic puzzles and reasoning problems
   - Customer service transcripts
   - Technical documentation and knowledge bases
   - Expert problem-solving sessions

2. **Curriculum Progression:**
   - Start with simple, single-step problems
   - Gradually increase complexity
   - Introduce edge cases and exceptions
   - Add time pressure and resource constraints

3. **Validation Requirements:**
   - Cross-validation by domain experts
   - Automated consistency checking
   - Performance benchmarking
   - Edge case coverage

4. **Usage Guidelines:**
   - Pre-process data to normalize formats
   - Implement data augmentation
   - Balance difficulty distribution
   - Ensure cultural/linguistic diversity

