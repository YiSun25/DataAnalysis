# Environment Setup

## PostgreSQL Deployment
PostgreSQL is deployed with Docker.

### Start PostgreSQL container
```powershell
docker start my-postgres
docker ps
```

### Stop PostgreSQL container
```powershell
docker stop my-postgres
```

## Connection Information

### Connection Details
- Host: `localhost`
- Port: `5432`
- Database: `mydb`
- Username: `postgres`
- Password: `123456`

## VS Code Setup

### Installed Extensions
- PostgreSQL (Microsoft)
- Python (Microsoft)
- Jupyter (Microsoft)

### Python Interpreter
`E:\softwares\Anaconda\Install\python.exe`

## Current Status

### Status Summary
- Docker is working
- PostgreSQL container is running correctly
- VS Code can connect to PostgreSQL
- Basic SQL queries can be executed successfully