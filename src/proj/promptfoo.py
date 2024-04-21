import json
from io import StringIO
from pathlib import Path
import pandas as pd


def read_results(model_folder: Path) -> pd.DataFrame:
    input_file = model_folder / "output" / "latest.json"
    assert input_file.exists(), f"File not found: {input_file}"
    
    data = pd.read_json(input_file)
    with StringIO(json.dumps(data.iloc[1, 0])) as f:
        results = pd.read_json(f)
    
    model = model_folder.name
    
    rows = []
    for _, r in results.iterrows():
        passed = r["success"]
        score_avg = r["score"]
        prompt = Path(r["prompt"]["display"]).stem
        test_description = r["vars"]["test_description"]
        
        for comp_result in r.loc["gradingResult"]["componentResults"]:
            rows.append({
                "model": model,
                "passed": passed,
                "score_avg": score_avg,
                "prompt": prompt,
                "test_description": test_description,
                "comp_pass": comp_result["pass"],
                "comp_score": comp_result["score"],
                "comp_reason": comp_result["reason"],
                "comp_desc": comp_result["assertion"]["value"],
                "comp_type": comp_result["assertion"]["type"],
            })
    
    return pd.DataFrame(rows)
