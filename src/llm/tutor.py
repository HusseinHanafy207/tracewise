"""LLM tutor layer — turns (student_state, intervention, language) into
natural-language output. This layer NEVER decides what the student needs;
it only communicates the decision already made by the student model +
policy. Keep that boundary explicit in the writeup.

Left as a stub with a clear interface so it can be filled in during
Day 11-12 without touching the ML core.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TutorRequest:
    student_state: dict          # e.g. {"skill": "backpropagation", "mastery": 0.37, ...}
    intervention: str            # e.g. "worked_example"
    misconception: Optional[str] # free-text, if detected
    language: str                # "en" or "ar"
    retrieved_context: List[str] # RAG snippets from course material


class LLMTutor:
    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-4-6"):
        self.provider = provider
        self.model = model

    def build_prompt(self, req: TutorRequest) -> str:
        lang_instruction = (
            "Respond in Arabic." if req.language == "ar" else "Respond in English."
        )
        context_block = "\n".join(req.retrieved_context) if req.retrieved_context else ""
        misconception_block = (
            f"Detected misconception: {req.misconception}\n" if req.misconception else ""
        )
        return f"""You are a patient STEM tutor. {lang_instruction}

Student state: {req.student_state}
{misconception_block}
Selected teaching intervention (already decided by the system, do not
override it): {req.intervention}

Relevant course material:
{context_block}

Produce a response appropriate to the intervention type above. If the
intervention is "worked_example", give a step-by-step worked example. If
it is "socratic_hint", ask a guiding question rather than giving the
answer. Keep it concise and specific to the student's stated gap."""

    def generate(self, req: TutorRequest) -> str:
        """Placeholder — wire up to the Anthropic API (see repo's
        `anthropic_api_in_artifacts` pattern) during Day 11-12. Raises
        until implemented so it fails loudly instead of silently mocking.
        """
        raise NotImplementedError(
            "Wire this up to an LLM API call during Day 11-12. "
            "Prompt is ready via self.build_prompt(req)."
        )
