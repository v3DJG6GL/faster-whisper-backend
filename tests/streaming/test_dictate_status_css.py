"""Cascade regression for /dictate's status colours.

NAV_CSS ships `header .subbar #status:not(.pill) { color: var(--dim); ... }`;
:not() takes the specificity of its argument, so that selector ties the
page's `header .subbar #status.live/.error` rules at (0,1,2,1) and source
order decides the cascade. The page rules must therefore sit AFTER the
{{NAV_CSS}} placeholder -- placed before it, the page's only success/failure
signal stayed --dim forever.
"""


def test_dictate_status_colours_win_the_cascade(client):
    html = client.get("/dictate").text
    # Anchor on the nav.css RULE: the page's own CSS comment above its rules
    # names the same selector (followed by a backtick, not " {"), and moving
    # comment and rules together above {{NAV_CSS}} would match it instead.
    assert html.count("#status:not(.pill) {") == 1
    shared = html.index("header .subbar #status:not(.pill) {")
    assert shared < html.index("header .subbar #status.live")
    assert shared < html.index("header .subbar #status.error")
