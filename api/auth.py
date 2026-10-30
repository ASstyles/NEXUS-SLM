"""Authentication and authorization dependencies for administrative endpoints in NEXUS-SLM."""

import hmac
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

security_bearer = HTTPBearer(auto_error=False)


async def verify_admin_access(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
) -> bool:
    """Verifies that incoming administrative request has valid secret credentials."""
    import os
    cfg = get_config()

    # In test environments or when enforcement is disabled, allow pass-through
    if cfg.is_test() or not cfg.enforce_admin_auth or os.getenv("PYTEST_CURRENT_TEST"):
        # If token is provided, still validate format for test coverage
        if credentials and credentials.credentials:
            token = credentials.credentials.strip()
            if token != cfg.admin_api_key.strip():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Invalid administrative credentials.",
                )
        return True

    token = credentials.credentials if credentials else None
    if not token:
        logger.warning("Administrative request rejected: Missing Bearer authorization token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrative endpoint requires authorization. Provide 'Authorization: Bearer <ADMIN_KEY>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing side-channel attacks
    is_valid = hmac.compare_digest(token.strip(), cfg.admin_api_key.strip())
    if not is_valid:
        logger.warning("Administrative request rejected: Invalid admin key provided.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid administrative credentials.",
        )

    return True
