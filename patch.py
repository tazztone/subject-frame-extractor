# 1. Update ui/tabs/subject_tab.py
with open("ui/tabs/subject_tab.py", "r") as f:
    content = f.read()

content = content.replace(
    'with gr.Tab("🔍 Scan Video for People"):', 'with gr.Tab("🔍 Scan Video for People", elem_id="scan_video_tab"):'
)
with open("ui/tabs/subject_tab.py", "w") as f:
    f.write(content)

# 2. Update tests/ui/ui_locators.py
with open("tests/ui/ui_locators.py", "r") as f:
    content = f.read()

if "SCAN_VIDEO_TAB" not in content:
    content = content.replace(
        'START_PRE_ANALYSIS = "#start_pre_analysis_button"',
        'START_PRE_ANALYSIS = "#start_pre_analysis_button"\n    SCAN_VIDEO_TAB = "#scan_video_tab"',
    )
    with open("tests/ui/ui_locators.py", "w") as f:
        f.write(content)

# 3. Update tests/ui/test_ui_interactions.py
with open("tests/ui/test_ui_interactions.py", "r") as f:
    content = f.read()

content = content.replace(
    'page.get_by_role("tab", name="Scan Video for People", exact=False).click()',
    "page.locator(Selectors.SCAN_VIDEO_TAB).click()",
)
with open("tests/ui/test_ui_interactions.py", "w") as f:
    f.write(content)

print("Patch applied successfully.")
