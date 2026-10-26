from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.models import User


async def test_committed_changes_are_rolled_back_after_the_test(
    db_session: AsyncSession, test_database_url: str
) -> None:
    db_session.add(User(email="rollback@example.com", hashed_password="x"))
    await db_session.commit()  # only releases a savepoint; the outer transaction stays open

    # A separate connection sees only truly committed data, so it must not see the row yet.
    other = create_async_engine(test_database_url, poolclass=NullPool)
    async with other.connect() as connection:
        count = await connection.scalar(text("SELECT count(*) FROM users"))
    await other.dispose()

    assert count == 0
