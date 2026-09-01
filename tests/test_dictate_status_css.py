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
    shared = html.index("#status:not(.pill)")
    assert shared < html.index("header .subbar #status.live")
    assert shared < html.index("header .subbar #status.error")
