def main():
    # Initialize state
    state = {
        'question_state': {
            'current': 'action',
            'answers': {
                'action': '',
                'role': '',
                'context': '',
                'response': '',
                'keywords': [],
                'expanded_actions': [],
                'expanded_roles': [],
                'expanded_contexts': [],
                'expanded_responses': []
            }
        },
        'prompt_history': [],
        'session_log': []
    }

    # Suggested options
    actions = [
        'compose', 'evaluate', 'generate', 'simulate', 'analyze', 'recommend',
        'design', 'summarize', 'explain', 'predict', 'translate', 'brainstorm',
        'modify', 'study', 'build', 'create', 'transcribe', 'edit', 'enhance',
        'debug', 'test', 'research'
    ]
    roles = [
        'educator', 'strategist', 'historian', 'ethicist', 'data scientist',
        'creative writer', 'engineer', 'consultant', 'researcher', 'marketer',
        'policy analyst', 'developer', 'artist', 'journalist', 'entrepreneur',
        'therapist', 'futurist', 'cartoonist', 'doctor', 'scientist', 'mechanic',
        'accountant', 'designer', 'mathematician'
    ]
    response_types = [
        'narrative', 'image', 'report', 'diagram', 'policy brief', 'code',
        'list', 'summary', 'plan', 'visualization', 'article', 'email',
        'meme', 'story', 'essay', 'resume', 'poem', 'infographic', 'chart',
        'website', 'sketch', 'digital art', 'cartoon', 'painting', 'program'
    ]
    visual_responses = ['image', 'cartoon', 'sketch', 'painting', 'chart']
    context_examples = {
        'educator': ['teaching a high school AI class', 'designing a STEM workshop'],
        'strategist': ['planning a corporate expansion', 'devising a marketing campaign'],
        'historian': ['analyzing 19th-century trade routes', 'researching ancient civilizations'],
        'ethicist': ['evaluating AI bias implications', 'assessing corporate ethics policies'],
        'data scientist': ['modeling customer churn', 'analyzing genomic data'],
        'creative writer': ['crafting a fantasy novel', 'writing a screenplay'],
        'engineer': ['optimizing a renewable energy system', 'designing a smart city infrastructure'],
        'consultant': ['advising on digital transformation', 'assessing supply chain efficiency'],
        'researcher': ['studying climate change impacts', 'investigating quantum computing'],
        'marketer': ['launching a social media campaign', 'analyzing consumer trends'],
        'policy analyst': ['drafting environmental regulations', 'evaluating healthcare policies'],
        'developer': ['building a web application', 'debugging a machine learning pipeline'],
        'artist': ['creating a mural for a community center', 'designing a digital art installation'],
        'journalist': ['investigating local government policies', 'writing a feature on tech startups'],
        'entrepreneur': ['pitching a sustainable startup idea', 'developing a business model'],
        'therapist': ['counseling on stress management', 'designing a mental health workshop'],
        'futurist': ['predicting trends in AI development', 'envisioning sustainable cities in 2050'],
        'cartoonist': ['creating a comic strip for a magazine', 'designing characters for an animated series'],
        'doctor': ['diagnosing a patient with chronic symptoms', 'developing a public health campaign']
    }
    stop_words = [
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with'
    ]
    common_errors = {'vague': ['detail', 'details', 'detailed']}

    def append_output(state, text, is_error=False):
        prefix = "[ERROR] " if is_error else "[OUTPUT] "
        print(prefix + text)
        state['session_log'].append({'type': 'error' if is_error else 'output', 'text': text})

    def show_help(state):
        help_text = """
WebXOS ARCR: How It Works
The ARCR system guides you to create precise AI prompts in four steps: Action, Role, Context, and Response.

1. **Action**: Choose what the AI should do (e.g., compose, analyze, modify, study).
2. **Role**: Define the AI's perspective (e.g., educator, developer, cartoonist, doctor).
3. **Context**: Provide specific details or scenarios (e.g., "teaching a class"). Be specific to ensure clarity.
4. **Response**: Select the output type (e.g., narrative, code, sketch, digital art).

After building your prompt, you can edit it to refine components or expand it by adding more actions, roles, contexts, or responses. Prompts are displayed in the console for copying. Type 'help' to see this guide, 'clear' to reset, or 'exit' to quit.
"""
        append_output(state, help_text)

    def update_progress(state):
        steps = ['Action', 'Role', 'Context', 'Response']
        current_index = ['action', 'role', 'context', 'response'].index(state['question_state']['current']) if state['question_state']['current'] in ['action', 'role', 'context', 'response'] else -1
        if current_index >= 0:
            return f"Step {current_index + 1}/4: {steps[current_index]}"
        elif state['question_state']['current'] == 'edit_prompt':
            return "Editing Prompt"
        elif state['question_state']['current'] == 'expand_prompt':
            return "Expanding Prompt"
        elif state['question_state']['current'].startswith('expand_'):
            component = state['question_state']['current'].replace('expand_', '')
            return f"Expanding {component.capitalize()}"
        return ""

    def ask_question(state):
        progress = update_progress(state)
        if progress:
            print(f"[PROGRESS] {progress}")
        if state['question_state']['current'] == 'action':
            append_output(state, f"Specify the action (e.g., compose: write content, analyze: examine data, generate: create new output): {', '.join(actions)}")
        elif state['question_state']['current'] == 'role':
            append_output(state, f"Define the role (e.g., educator: teaches, developer: codes, artist: creates visuals): {', '.join(roles)}")
        elif state['question_state']['current'] == 'context':
            role = state['question_state']['answers']['role'].lower()
            examples = context_examples.get(role, ['in a specific scenario', 'with clear objectives'])
            append_output(state, f"Describe the context in detail (examples: {', '.join(examples)}):")
        elif state['question_state']['current'] == 'response':
            append_output(state, f"Choose the response type (e.g., narrative: story, image: visual, code: program): {', '.join(response_types)}")
        elif state['question_state']['current'] == 'expand_prompt':
            append_output(state, "Do you want to expand your prompt? (yes, no)")
        elif state['question_state']['current'] == 'expand_select':
            append_output(state, "What do you want to expand? (action, role, context, response)")
        elif state['question_state']['current'] == 'expand_action':
            append_output(state, f"Specify the additional action: {', '.join(actions)}")
        elif state['question_state']['current'] == 'expand_role':
            append_output(state, f"Define the additional role: {', '.join(roles)}")
        elif state['question_state']['current'] == 'expand_context':
            role = state['question_state']['answers']['role'].lower()
            examples = context_examples.get(role, ['in a specific scenario', 'with clear objectives'])
            append_output(state, f"Describe the additional context (examples: {', '.join(examples)}):")
        elif state['question_state']['current'] == 'expand_response':
            append_output(state, f"Choose the additional response type: {', '.join(response_types)}")

    def validate_answer(state, input_str):
        normalized_input = input_str.strip().lower()
        if not input_str.strip():
            append_output(state, "Error: Input cannot be empty.", True)
            return False
        if state['question_state']['current'] == 'action' and normalized_input not in actions:
            append_output(state, f"Error: Select from: {', '.join(actions)}", True)
            return False
        elif state['question_state']['current'] == 'role' and normalized_input not in roles:
            append_output(state, f"Error: Select from: {', '.join(roles)}", True)
            return False
        elif state['question_state']['current'] == 'context' and len(input_str.strip()) < 10:
            append_output(state, "Error: Context must be specific (10+ characters).", True)
            return False
        elif state['question_state']['current'] == 'response' and normalized_input not in response_types:
            append_output(state, f"Error: Select from: {', '.join(response_types)}", True)
            return False
        elif state['question_state']['current'] == 'expand_prompt' and normalized_input not in ['yes', 'no']:
            append_output(state, "Error: Please enter 'yes' or 'no'.", True)
            return False
        elif state['question_state']['current'] == 'expand_select' and normalized_input not in ['action', 'role', 'context', 'response']:
            append_output(state, "Error: Select from: action, role, context, response", True)
            return False
        elif state['question_state']['current'] == 'expand_action' and normalized_input not in actions:
            append_output(state, f"Error: Select from: {', '.join(actions)}", True)
            return False
        elif state['question_state']['current'] == 'expand_role' and normalized_input not in roles:
            append_output(state, f"Error: Select from: {', '.join(roles)}", True)
            return False
        elif state['question_state']['current'] == 'expand_context' and len(input_str.strip()) < 10:
            append_output(state, "Error: Additional context must be specific (10+ characters).", True)
            return False
        elif state['question_state']['current'] == 'expand_response' and normalized_input not in response_types:
            append_output(state, f"Error: Select from: {', '.join(response_types)}", True)
            return False
        for term in common_errors['vague']:
            if term in normalized_input:
                append_output(state, "Error: Avoid vague terms (e.g., 'detail').", True)
                return False
        return True

    def autocorrect_input(input_str):
        corrected = input_str
        corrected = corrected.replace('summary', 'summary', 1).replace('SUMMARY', 'summary', 1)
        corrected = corrected.replace('explain', 'explain', 1).replace('EXPLAIN', 'explain', 1)
        for term in common_errors['vague']:
            corrected = corrected.replace(term, 'specific')
        return corrected

    def extract_keywords(input_str):
        import re
        words = re.split(r'[.,!?;:\'"\(\)\s]+', input_str.lower())
        keywords = []
        for word in words:
            if word and word not in stop_words and len(word) > 2 and word not in keywords:
                keywords.append(word)
        return keywords

    def generate_prompt(state):
        answers = state['question_state']['answers']
        prompt = (
            f"The user has a request structured as Action > Role > Context > Response.\n"
            f"As a {answers['role']}, perform the following task:\n"
            f"- **Action**: {answers['action']}\n"
            f"- **Context**: {answers['context']}\n"
            f"- **Keywords for DeepSearch**: {', '.join(answers['keywords']) if answers['keywords'] else 'none'} (use these to pretrain on relevant web data before processing)\n"
            f"- **Response**: Deliver a {answers['response']}\n"
        )
        if answers['response'].lower() in visual_responses:
            prompt += (
                f"\n**Visual Generation Instructions**: User requests visual generation. Use DeepSearch to gather relevant visual data and apply thinking logic to ensure high-quality output.\n"
            )
        has_expansions = (
            len(answers['expanded_actions']) > 0 or
            len(answers['expanded_roles']) > 0 or
            len(answers['expanded_contexts']) > 0 or
            len(answers['expanded_responses']) > 0
        )
        if has_expansions:
            prompt += "\nExpanded Components:\n"
            if answers['expanded_actions']:
                prompt += f"- **Additional Actions**: {', '.join(answers['expanded_actions'])}\n"
            if answers['expanded_roles']:
                prompt += f"- **Additional Roles**: {', '.join(answers['expanded_roles'])}\n"
            if answers['expanded_contexts']:
                prompt += f"- **Additional Contexts**: {', '.join(answers['expanded_contexts'])}\n"
            if answers['expanded_responses']:
                prompt += f"- **Additional Responses**: {', '.join(answers['expanded_responses'])}\n"
        prompt += "Generated by WebXOS ARCR 2025 (Copyright © 2025)."
        return prompt

    def update_history(state, prompt):
        state['prompt_history'].insert(0, {'prompt': prompt})
        if len(state['prompt_history']) > 5:
            state['prompt_history'].pop()
        append_output(state, "Prompt saved.")
        state['session_log'].append({'type': 'output', 'text': 'Prompt saved.'})

    def clear_console(state):
        state['question_state'] = {
            'current': 'action',
            'answers': {
                'action': '',
                'role': '',
                'context': '',
                'response': '',
                'keywords': [],
                'expanded_actions': [],
                'expanded_roles': [],
                'expanded_contexts': [],
                'expanded_responses': []
            }
        }
        state['prompt_history'] = []
        state['session_log'] = []
        append_output(state, "Console and state cleared. Ready for a new prompt.")
        ask_question(state)

    def edit_prompt(state):
        state['question_state']['current'] = 'edit_prompt'
        while True:
            prompt = generate_prompt(state)
            append_output(state, f"Generated Prompt:\n\n---\n{prompt}\n---")
            append_output(state, "Are you satisfied with this prompt? (yes/no)")
            satisfied = input("> ").strip().lower()
            if satisfied == 'yes':
                update_history(state, prompt)
                append_output(state, "Prompt finalized! (Note: Copy to clipboard not supported in this environment; please copy the prompt above.)")
                break
            elif satisfied == 'no':
                append_output(state, "Which part do you want to edit? (action/role/context/response)")
                part = input("> ").strip().lower()
                if part in ['action', 'role', 'context', 'response']:
                    if part == 'context':
                        append_output(state, f"Enter new context (must be specific, 10+ characters):")
                        new_value = input("> ").strip()
                        if len(new_value) < 10:
                            append_output(state, "Error: Context must be at least 10 characters.", True)
                            continue
                        state['question_state']['answers']['context'] = new_value
                        state['question_state']['answers']['keywords'] = extract_keywords(new_value)
                    else:
                        valid_options = actions if part == 'action' else roles if part == 'role' else response_types
                        append_output(state, f"Enter new {part} (options: {', '.join(valid_options)}):")
                        new_value = input("> ").strip().lower()
                        if part == 'action' and new_value not in actions:
                            append_output(state, f"Error: Select from: {', '.join(actions)}", True)
                            continue
                        elif part == 'role' and new_value not in roles:
                            append_output(state, f"Error: Select from: {', '.join(roles)}", True)
                            continue
                        elif part == 'response' and new_value not in response_types:
                            append_output(state, f"Error: Select from: {', '.join(response_types)}", True)
                            continue
                        state['question_state']['answers'][part] = new_value
                else:
                    append_output(state, "Invalid part selected.", True)
            else:
                append_output(state, "Invalid input. Please enter 'yes' or 'no'.", True)
        state['question_state']['current'] = 'expand_prompt'
        ask_question(state)

    def handle_input(state):
        print("[INFO] Welcome to WebXOS ARCR! Create precise AI prompts using the ARCR system (Action > Role > Context > Response). Enter each component to build a structured prompt.")
        state['session_log'].append({'type': 'output', 'text': "Welcome to WebXOS ARCR! Create precise AI prompts using the ARCR system (Action > Role > Context > Response). Enter each component to build a structured prompt."})
        ask_question(state)
        while True:
            try:
                input_str = input("> ").strip()
                if input_str.lower() == 'clear':
                    clear_console(state)
                    continue
                elif input_str.lower() == 'help':
                    show_help(state)
                    ask_question(state)
                    continue
                elif input_str.lower() == 'exit':
                    append_output(state, "Exiting ARCR.")
                    break

                input_str = autocorrect_input(input_str)
                if not validate_answer(state, input_str):
                    continue
                append_output(state, f"> {input_str}")
                state['session_log'].append({'type': 'input', 'text': input_str})

                if state['question_state']['current'] == 'action':
                    state['question_state']['answers']['action'] = input_str
                    state['question_state']['current'] = 'role'
                    ask_question(state)
                elif state['question_state']['current'] == 'role':
                    state['question_state']['answers']['role'] = input_str
                    state['question_state']['current'] = 'context'
                    ask_question(state)
                elif state['question_state']['current'] == 'context':
                    state['question_state']['answers']['context'] = input_str
                    state['question_state']['answers']['keywords'] = extract_keywords(input_str)
                    state['question_state']['current'] = 'response'
                    ask_question(state)
                elif state['question_state']['current'] == 'response':
                    state['question_state']['answers']['response'] = input_str
                    state['question_state']['current'] = 'edit_prompt'
                    edit_prompt(state)
                elif state['question_state']['current'] == 'expand_prompt':
                    if input_str.lower() == 'yes':
                        state['question_state']['current'] = 'expand_select'
                    else:
                        append_output(state, "Prompt expansion skipped.")
                        break
                    ask_question(state)
                elif state['question_state']['current'] == 'expand_select':
                    if input_str.lower() in ['action', 'role', 'context', 'response']:
                        state['question_state']['current'] = f"expand_{input_str.lower()}"
                    else:
                        append_output(state, "Error: Select from: action, role, context, response", True)
                        continue
                    ask_question(state)
                elif state['question_state']['current'] == 'expand_action':
                    state['question_state']['answers']['expanded_actions'].append(input_str)
                    prompt = generate_prompt(state)
                    append_output(state, f"Updated Prompt:\n\n---\n{prompt}\n---")
                    update_history(state, prompt)
                    append_output(state, "Prompt updated! (Note: Copy to clipboard not supported in this environment; please copy the prompt above.)")
                    state['question_state']['current'] = 'expand_prompt'
                    ask_question(state)
                elif state['question_state']['current'] == 'expand_role':
                    state['question_state']['answers']['expanded_roles'].append(input_str)
                    prompt = generate_prompt(state)
                    append_output(state, f"Updated Prompt:\n\n---\n{prompt}\n---")
                    update_history(state, prompt)
                    append_output(state, "Prompt updated! (Note: Copy to clipboard not supported in this environment; please copy the prompt above.)")
                    state['question_state']['current'] = 'expand_prompt'
                    ask_question(state)
                elif state['question_state']['current'] == 'expand_context':
                    state['question_state']['answers']['expanded_contexts'].append(input_str)
                    prompt = generate_prompt(state)
                    append_output(state, f"Updated Prompt:\n\n---\n{prompt}\n---")
                    update_history(state, prompt)
                    append_output(state, "Prompt updated! (Note: Copy to clipboard not supported in this environment; please copy the prompt above.)")
                    state['question_state']['current'] = 'expand_prompt'
                    ask_question(state)
                elif state['question_state']['current'] == 'expand_response':
                    state['question_state']['answers']['expanded_responses'].append(input_str)
                    prompt = generate_prompt(state)
                    append_output(state, f"Updated Prompt:\n\n---\n{prompt}\n---")
                    update_history(state, prompt)
                    append_output(state, "Prompt updated! (Note: Copy to clipboard not supported in this environment; please copy the prompt above.)")
                    state['question_state']['current'] = 'expand_prompt'
                    ask_question(state)
            except Exception as e:
                append_output(state, f"Error processing input: {str(e)}", True)

    handle_input(state)

main()
