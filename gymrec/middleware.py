"""Small deployment middleware."""


class FrameAncestorsMiddleware:
    """Allow huggingface.co to show the app in its Space page iframe (instead of X-Frame-Options: DENY)."""

    POLICY = "frame-ancestors 'self' https://huggingface.co https://*.hf.space"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response.setdefault("Content-Security-Policy", self.POLICY)
        return response
