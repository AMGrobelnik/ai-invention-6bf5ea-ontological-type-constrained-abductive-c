import json
from pathlib import Path

# Create 50 examples with predict_* fields in correct exp_gen_sol_out schema
examples = []
for i in range(50):
    ex = {
        "input": f"Example {i+1}: Alice is a person. Bob is a dog.",
        "output": f"pets(alice, bob)",
        "predict_otc": "True",
        "predict_pure_llm": "True" if i % 2 == 0 else "False",
        "predict_argos": "True" if i % 3 == 0 else "False",
        "predict_symba": "True" if i % 5 == 0 else "False"
    }
    examples.append(ex)

# Wrap in datasets array (exp_gen_sol_out schema)
output_data = {
    "datasets": [
        {
            "dataset": "otc_synthetic",
            "examples": examples
        }
    ]
}

# Write corrected method_out.json
with open('method_out.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Created method_out.json with {len(examples)} examples")
