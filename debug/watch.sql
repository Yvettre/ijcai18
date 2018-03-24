SELECT context_time, user_id, item_id, item_category_list, item_brand_id
	, shop_id, istrade
FROM train
WHERE user_id = (
	SELECT user_id
	FROM train
	WHERE instance_id = 8815105589080333699
)
ORDER BY context_time;