
    with get_session() as session:
        try:
            query = session.query(UserConversation)

            filtered_query = (
                else query.filter_by(user_id=user_id)
            )

            user_conversations = filtered_query.all()

