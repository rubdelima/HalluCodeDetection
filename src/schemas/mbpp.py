from pydantic import BaseModel
import re

class MBPPExample(BaseModel):
    benchmark_id: int
    prompt: str
    tests: list[str]
    function_name: str | None
    function_definition: str | None
    function_signature: str | None
    
    @classmethod
    def from_dataset(cls, example)-> 'MBPPExample':
        code = example['code']
        match = re.search(r'(def\s+\w+\s*\(.*?\)\s*:)', code, re.DOTALL)
        function_definition = match.group(1).strip() if match else None

        name_match = re.search(r'def\s+(\w+)\s*\(', function_definition or "")
        function_name = name_match.group(1) if name_match else None

        # Keep only the signature line (single-line form) for prompts
        signature_match = re.search(r'(def\s+\w+\s*\(.*?\)\s*:)', function_definition or "")
        function_signature = signature_match.group(1).replace("\n", " ").strip() if signature_match else None
        
        if not function_name:
            print(f"Could not extract function name from code: {code}")
        
        return cls(
            benchmark_id=example['task_id'],
            prompt=example['prompt'],
            tests=example['test_list'],
            function_name=function_name,
            function_definition=function_definition,
            function_signature=function_signature
        )
    
    @property
    def benchmark_name(self)->str:
        return f"mbpp"