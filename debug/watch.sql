SELECT context_time, user_id, item_id, item_brand_id, item_price_level
	, shop_id, istrade
FROM train
WHERE user_id = (
	SELECT user_id
	FROM train
	WHERE instance_id = 7617268014077556154
)
ORDER BY context_time;

SELECT user_id, user_gender_id, user_age_level, user_occupation_id
FROM train
WHERE user_id = (
	SELECT user_id
	FROM train
	WHERE instance_id = 7617268014077556154
)
GROUP BY user_id, user_gender_id, user_age_level, user_occupation_id;