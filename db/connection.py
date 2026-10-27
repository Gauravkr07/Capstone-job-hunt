from collections.abc import AsyncGenerator

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import DATABASE_URL
from db.models import Base, Skill


engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # log SQL statements to the console when True
    future=True,  # use SQLAlchemy 2.0 style behavior
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,  # seconds to wait for a connection from the pool
    pool_recycle=1800,  # seconds to recycle connections and avoid stale connections
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


# create_all() only creates missing *tables*; columns added to existing tables
# need an explicit, idempotent ALTER. Append new ones here -- never edit old ones.
_COLUMN_MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
    "ALTER TABLE user_details ADD COLUMN IF NOT EXISTS resume_source VARCHAR(20)",
    "ALTER TABLE user_details ADD COLUMN IF NOT EXISTS resume_filename VARCHAR(500)",
    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS posted_by VARCHAR(150) REFERENCES users(username)",
    "CREATE INDEX IF NOT EXISTS ix_jobs_posted_by ON jobs (posted_by)",
    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS candidate VARCHAR(150)",
    "CREATE INDEX IF NOT EXISTS ix_chat_messages_candidate ON chat_messages (candidate)",
    "ALTER TABLE chat_leads ADD COLUMN IF NOT EXISTS candidate VARCHAR(150)",
    "CREATE INDEX IF NOT EXISTS ix_chat_leads_candidate ON chat_leads (candidate)",
]


async def init_db() -> None:
    """Create missing tables, then apply the idempotent column migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for statement in _COLUMN_MIGRATIONS:
            await conn.execute(text(statement))


async def seed_skills() -> None:
    """Populate the skills table from the SKILL_KEYWORDS taxonomy, idempotently."""
    from helpers.constant import SKILL_KEYWORDS

    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Skill.name))
        existing_names = {row[0] for row in existing.all()}

        for category, skill_group in SKILL_KEYWORDS.items():
            if not isinstance(skill_group, dict):
                continue
            for skill_name in skill_group:
                if skill_name not in existing_names:
                    session.add(Skill(name=skill_name, category=category))
                    existing_names.add(skill_name)

        await session.commit()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async dependency that yields a session from the connection pool."""
    async with AsyncSessionLocal() as session:
        yield session
