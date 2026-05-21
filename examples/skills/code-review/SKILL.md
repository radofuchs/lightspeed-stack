---
name: code-review
description: Review code changes for quality, security, and maintainability. Use when a user asks for a code review, wants feedback on their code, or asks about best practices for a code change.
---

# Code Review

## When to use this skill

Use this skill when:
- A user asks you to review code or a diff
- A user wants feedback on code quality
- A user asks about best practices for a specific change

## Review checklist

### Correctness
- Does the code do what it claims to do?
- Are edge cases handled (empty inputs, null values, boundary conditions)?
- Are error conditions handled appropriately?

### Security
- Is user input validated and sanitized?
- Are secrets hardcoded or properly managed via environment variables?
- Are SQL queries parameterized (no string concatenation)?
- Are file paths validated to prevent directory traversal?

### Maintainability
- Are variable and function names descriptive?
- Is the code structured for readability (appropriate function length, single responsibility)?
- Are there comments explaining non-obvious logic?
- Is there unnecessary complexity that could be simplified?

### Performance
- Are there obvious performance issues (N+1 queries, unnecessary loops, missing indexes)?
- Are large data sets handled efficiently (pagination, streaming)?
- Are expensive operations cached where appropriate?

### Testing
- Are there tests for new functionality?
- Do tests cover edge cases and error conditions?
- Are tests readable and well-structured?

## Review format

Structure your review as follows:

1. **Summary**: One sentence describing the overall change
2. **Strengths**: What the code does well (be specific)
3. **Issues**: Problems that should be fixed, ordered by severity
   - **Critical**: Bugs, security issues, data loss risks
   - **Major**: Logic errors, missing error handling
   - **Minor**: Style, naming, documentation
4. **Suggestions**: Optional improvements that are not blocking
