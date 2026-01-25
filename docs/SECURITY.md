# Security Guide

This document outlines security considerations, best practices, and implementation details for the AI-Generated Voice Detection API.

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication and Authorization](#authentication-and-authorization)
3. [API Key Management](#api-key-management)
4. [Input Validation and Sanitization](#input-validation-and-sanitization)
5. [Data Protection](#data-protection)
6. [Network Security](#network-security)
7. [Error Handling and Information Disclosure](#error-handling-and-information-disclosure)
8. [Logging and Monitoring](#logging-and-monitoring)
9. [Deployment Security](#deployment-security)
10. [Security Testing](#security-testing)
11. [Incident Response](#incident-response)

## Security Overview

The AI-Generated Voice Detection API implements multiple layers of security to protect against common threats and ensure secure operation in production environments.

### Security Principles

1. **Defense in Depth**: Multiple security layers and controls
2. **Least Privilege**: Minimal access rights and permissions
3. **Fail Secure**: Secure defaults and fail-safe mechanisms
4. **Security by Design**: Security considerations integrated from the start
5. **Zero Trust**: Verify all requests and inputs

### Threat Model

**Identified Threats:**

- Unauthorized API access
- Malicious audio file uploads
- Data injection attacks
- Information disclosure
- Denial of Service (DoS) attacks
- Man-in-the-middle attacks
- API key compromise

## Authentication and Authorization

### API Key Authentication

The API uses API key-based authentication for all protected endpoints.

**Implementation:**

- API keys are validated on every request to protected endpoints
- Keys are passed in the `x-api-key` HTTP header
- Invalid or missing keys result in 401 Unauthorized responses

**Security Features:**

- Case-sensitive key validation
- Secure key storage in environment variables
- No key logging or exposure in error messages
- Support for multiple active keys (key rotation)

### Protected Endpoints

| Endpoint                    | Authentication Required |
| --------------------------- | ----------------------- |
| `POST /api/voice-detection` | Yes                     |
| `GET /health`               | No                      |
| `GET /health/detailed`      | No                      |
| `GET /docs`                 | No                      |
| `GET /`                     | No                      |

### Authentication Bypass Prevention

- Middleware validates authentication before request processing
- No authentication bypass mechanisms
- OPTIONS requests (CORS preflight) are allowed without authentication
- All other requests to protected endpoints require valid API keys

## API Key Management

### Key Generation

**Recommended practices:**

- Use cryptographically secure random generators
- Minimum 32 characters length
- Include alphanumeric characters and symbols
- Avoid predictable patterns

**Example secure key generation:**

```bash
# Using OpenSSL
openssl rand -base64 32

# Using Python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Key Storage

**Environment Variables:**

```bash
# Development
API_KEYS=dev-api-key-12345,dev-backup-key-67890

# Production (use secure values)
API_KEYS=prod-secure-key-abc123xyz789,prod-backup-key-def456uvw012
```

**Best Practices:**

- Store keys in environment variables, not in code
- Use secrets management systems in production (AWS Secrets Manager, Azure Key Vault, etc.)
- Never commit keys to version control
- Use different keys for different environments

### Key Rotation

**Rotation Strategy:**

1. Generate new API key
2. Add new key to `API_KEYS` environment variable
3. Update clients to use new key
4. Remove old key after transition period
5. Monitor for any remaining usage of old key

**Automated Rotation:**

```bash
# Example rotation script
#!/bin/bash
NEW_KEY=$(openssl rand -base64 32)
echo "New API key generated: $NEW_KEY"
# Update environment configuration
# Notify clients of key change
# Schedule old key removal
```

### Key Compromise Response

**Immediate Actions:**

1. Remove compromised key from `API_KEYS`
2. Generate and deploy new key
3. Update all legitimate clients
4. Monitor for unauthorized usage
5. Review access logs for suspicious activity

## Input Validation and Sanitization

### Request Validation

**Pydantic Models:**

- All request data is validated using Pydantic models
- Type checking and format validation
- Custom validators for specific fields
- Automatic error responses for invalid data

**Validation Rules:**

- `language`: Must be one of the supported languages
- `audioFormat`: Must be "mp3"
- `audioBase64`: Must be valid base64 encoding
- Request size limits enforced

### Audio Data Validation

**Base64 Validation:**

- Verify valid base64 encoding
- Check for malicious characters or patterns
- Validate decoded data size limits

**MP3 Format Validation:**

- Verify MP3 file headers and structure
- Check for embedded malicious content
- Validate audio duration limits (0.5s - 300s)
- File size limits (max 50MB)

**Security Measures:**

```python
# Example validation implementation
def validate_audio_data(base64_data: str) -> bool:
    try:
        # Decode base64
        audio_bytes = base64.b64decode(base64_data)

        # Check size limits
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise ValueError("Audio file too large")

        # Validate MP3 format
        if not is_valid_mp3(audio_bytes):
            raise ValueError("Invalid MP3 format")

        return True
    except Exception:
        return False
```

### SQL Injection Prevention

- No direct SQL queries in the current implementation
- If database integration is added, use parameterized queries
- ORM usage with proper escaping
- Input sanitization for any database operations

### Command Injection Prevention

- No direct system command execution with user input
- If system commands are needed, use safe parameter passing
- Whitelist allowed characters and patterns
- Avoid shell interpretation of user data

## Data Protection

### Data in Transit

**HTTPS Enforcement:**

- TLS 1.2+ required for production
- Strong cipher suites configuration
- HSTS headers for HTTPS enforcement
- Certificate validation and management

**Configuration:**

```bash
# Production SSL settings
SSL_ENABLED=true
SSL_CERT_FILE=/app/certs/cert.pem
SSL_KEY_FILE=/app/certs/key.pem
```

### Data at Rest

**Audio Data:**

- Audio data is not persisted beyond request processing
- Temporary processing data is cleared after response
- No long-term storage of user audio content

**Configuration Data:**

- Sensitive configuration stored in environment variables
- SSL certificates stored with restricted file permissions
- Log files protected with appropriate access controls

**File Permissions:**

```bash
# Secure file permissions
chmod 600 .env                    # Environment file
chmod 600 certs/key.pem          # Private key
chmod 644 certs/cert.pem         # Certificate
chmod 700 logs/                  # Log directory
```

### Data Minimization

- Only required data is processed
- No unnecessary data collection or storage
- Automatic cleanup of temporary processing data
- No user tracking or profiling

## Network Security

### CORS Configuration

**Development:**

```python
# Permissive for development
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

**Production:**

```python
# Restrictive for production
CORS_ORIGINS=["https://yourdomain.com", "https://api.yourdomain.com"]
```

### Security Headers

**Implemented Headers:**

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Referrer-Policy: strict-origin-when-cross-origin`

**Nginx Configuration:**

```nginx
# Security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header Referrer-Policy "strict-origin-when-cross-origin";
```

### Rate Limiting

**Implementation:**

- Request rate limiting per API key
- Burst protection for rapid requests
- IP-based rate limiting as backup
- Configurable limits per environment

**Nginx Rate Limiting:**

```nginx
# Rate limiting configuration
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req zone=api burst=20 nodelay;
```

### Firewall Configuration

**Recommended Rules:**

- Allow only necessary ports (80, 443, 8000)
- Block direct access to internal services
- Implement IP whitelisting for admin access
- Log and monitor connection attempts

## Error Handling and Information Disclosure

### Secure Error Messages

**Principles:**

- No sensitive information in error messages
- Generic error messages for security-related failures
- Detailed errors only in development mode
- Consistent error response format

**Implementation:**

```python
# Secure error handling
try:
    # Process request
    pass
except AuthenticationError:
    # Generic authentication error
    return {"status": "error", "message": "Authentication failed"}
except ValidationError as e:
    # Safe validation error details
    return {"status": "error", "message": f"Validation error: {safe_message(e)}"}
except Exception:
    # Generic system error
    logger.error("Unexpected error", exc_info=True)
    return {"status": "error", "message": "Internal server error"}
```

### Information Leakage Prevention

- No stack traces in production responses
- No internal paths or system information exposed
- No database schema or internal structure details
- Sanitized error messages for external consumption

### Debug Mode Security

**Development vs Production:**

```python
# Development: Detailed errors
if settings.debug:
    return {"error": str(exception), "traceback": traceback.format_exc()}

# Production: Generic errors
else:
    return {"error": "Internal server error"}
```

## Logging and Monitoring

### Security Logging

**Logged Events:**

- Authentication attempts (success/failure)
- Authorization failures
- Input validation failures
- Suspicious request patterns
- System errors and exceptions
- Configuration changes

**Log Format:**

```python
# Security event logging
logger.info(
    "Authentication attempt",
    extra={
        "event": "auth_attempt",
        "ip": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "success": False,
        "reason": "invalid_key"
    }
)
```

### Log Security

**Protection Measures:**

- Restricted file permissions (600/640)
- Log rotation to prevent disk exhaustion
- No sensitive data in logs (API keys, audio content)
- Secure log transmission if using external systems

**Configuration:**

```python
# Secure logging configuration
LOG_FILE=logs/app-prod.log
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5
```

### Monitoring and Alerting

**Key Metrics:**

- Failed authentication attempts
- Unusual request patterns
- Error rates and types
- Response time anomalies
- Resource usage spikes

**Alert Conditions:**

- Multiple failed authentication attempts from same IP
- Unusual request volume or patterns
- High error rates
- System resource exhaustion
- SSL certificate expiration

## Deployment Security

### Container Security

**Docker Best Practices:**

- Use non-root user in containers
- Minimal base images (slim/alpine)
- Regular image updates and vulnerability scanning
- Read-only file systems where possible
- Resource limits and constraints

**Dockerfile Security:**

```dockerfile
# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set secure permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
```

### Environment Security

**Production Checklist:**

- [ ] Debug mode disabled
- [ ] Strong API keys configured
- [ ] HTTPS enabled with valid certificates
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Firewall rules configured
- [ ] Log monitoring setup
- [ ] Regular security updates scheduled

### Infrastructure Security

**Network Security:**

- Private networks for internal communication
- Load balancer with SSL termination
- Web Application Firewall (WAF) if applicable
- DDoS protection services

**Access Control:**

- SSH key-based authentication
- Multi-factor authentication for admin access
- Regular access review and cleanup
- Principle of least privilege

## Security Testing

### Automated Security Testing

**Static Analysis:**

- Code security scanning (bandit for Python)
- Dependency vulnerability scanning
- Configuration security checks
- Secret detection in code

**Dynamic Testing:**

- API security testing
- Authentication bypass testing
- Input validation testing
- Error handling verification

### Manual Security Testing

**Penetration Testing:**

- Authentication and authorization testing
- Input validation and injection testing
- Error handling and information disclosure
- Business logic security testing

**Security Review Checklist:**

- [ ] Authentication mechanisms
- [ ] Input validation coverage
- [ ] Error handling security
- [ ] Logging and monitoring
- [ ] Configuration security
- [ ] Dependency security

### Vulnerability Management

**Process:**

1. Regular vulnerability scanning
2. Risk assessment and prioritization
3. Patch management and deployment
4. Verification and testing
5. Documentation and reporting

## Incident Response

### Security Incident Types

**Potential Incidents:**

- API key compromise
- Unauthorized access attempts
- Data breach or exposure
- Service disruption attacks
- Malicious file uploads

### Response Procedures

**Immediate Response:**

1. Identify and contain the incident
2. Assess the scope and impact
3. Implement immediate countermeasures
4. Document all actions taken
5. Notify relevant stakeholders

**API Key Compromise Response:**

```bash
# Emergency key rotation
1. Remove compromised key from API_KEYS
2. Generate new emergency key
3. Update critical clients immediately
4. Monitor for continued unauthorized access
5. Investigate compromise source
```

### Recovery Procedures

**System Recovery:**

1. Restore from secure backups if needed
2. Apply security patches and updates
3. Verify system integrity
4. Gradual service restoration
5. Enhanced monitoring during recovery

### Post-Incident Activities

**Follow-up Actions:**

1. Detailed incident analysis
2. Root cause identification
3. Security control improvements
4. Process and procedure updates
5. Staff training and awareness

## Security Compliance

### Standards and Frameworks

**Applicable Standards:**

- OWASP API Security Top 10
- NIST Cybersecurity Framework
- ISO 27001 (if required)
- Industry-specific regulations

### Regular Security Activities

**Monthly:**

- Security log review
- Vulnerability scanning
- Access review and cleanup
- Security metrics analysis

**Quarterly:**

- Security control testing
- Incident response plan review
- Security training updates
- Third-party security assessments

**Annually:**

- Comprehensive security audit
- Penetration testing
- Risk assessment update
- Security policy review

## Contact Information

For security-related issues or questions:

- **Security Team**: [security@yourcompany.com]
- **Emergency Contact**: [emergency@yourcompany.com]
- **Bug Bounty Program**: [bugbounty@yourcompany.com] (if applicable)

**Responsible Disclosure:**
We encourage responsible disclosure of security vulnerabilities. Please contact our security team before publicly disclosing any issues.
