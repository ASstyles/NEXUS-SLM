"""Authentication and authorization dependencies for administrative endpoints in NEXUS-SLM."""

import hmac
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

security_bearer = HTTPBearer(auto_error=False)


async def verify_admin_access(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
) -> bool:
    """
    Verifies that incoming administrative request has valid secret credentials.
    Production authentication logic remains strict and consistent across all environments.
    No automatic bypasses based on test runner environment variables.
    """
    cfg = get_config()

    # When administrative authentication is explicitly disabled by configuration
    if not cfg.enforce_admin_auth:
        return True

    if not credentials or not credentials.credentials:
        logger.warning("Administrative request rejected: Missing Bearer authorization token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrative endpoint requires authorization. Provide 'Authorization: Bearer <ADMIN_KEY>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    configured_key = cfg.admin_api_key.strip()

    # Constant-time comparison to prevent timing side-channel attacks
    is_valid = hmac.compare_digest(token, configured_key)
    if not is_valid:
        logger.warning("Administrative request rejected: Invalid admin key provided.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid administrative credentials.",
        )

    return True
