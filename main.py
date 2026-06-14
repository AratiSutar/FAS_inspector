import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["FLAGS_call_stack_level"] = "0"

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level="info",
    )