create table user_histories
(
    id         bigint,
    user_id    bigint,
    post_id    bigint,
    created_at timestamp,
    updated_at timestamp
);

alter table user_histories
    owner to postgres;

INSERT INTO public.user_histories (id, user_id, post_id, created_at, updated_at) VALUES (1, 431, 3176, '2022-09-17 15:09:13.000000', '2022-09-17 15:09:16.000000');
INSERT INTO public.user_histories (id, user_id, post_id, created_at, updated_at) VALUES (2, 431, 3083, '2022-09-17 15:14:06.000000', '2022-09-17 15:14:09.000000');
INSERT INTO public.user_histories (id, user_id, post_id, created_at, updated_at) VALUES (null, 431, 3090, '2022-09-17 15:09:13.000000', '2022-09-17 15:09:16.000000');
INSERT INTO public.user_histories (id, user_id, post_id, created_at, updated_at) VALUES (null, 431, 2752, '2022-09-17 15:14:06.000000', '2022-09-17 15:14:09.000000');