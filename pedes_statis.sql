DROP TABLE IF EXISTS "pedes_statis";
CREATE TABLE "pedes_statis" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "total_count" integer DEFAULT 0,
  "area_a" integer,
  "area_b" integer DEFAULT 0,
  "area_c" integer DEFAULT 0,
  "area_d" integer DEFAULT 0,
  "q1_count" integer DEFAULT 0,
  "q2_count" integer DEFAULT 0,
  "q3_count" integer DEFAULT 0,
  "q4_count" integer DEFAULT 0,
  "q5_count" integer DEFAULT 0,
  "a0_count" integer DEFAULT 0,
  "a1_count" integer DEFAULT 0,
  "a2_count" integer DEFAULT 0,
  "a3_count" integer DEFAULT 0,
  "a4_count" integer DEFAULT 0,
  "a5_count" integer DEFAULT 0,
  "a6_count" integer DEFAULT 0,
  "a7_count" integer DEFAULT 0,
  "a8_count" integer DEFAULT 0,
  "a9_count" integer DEFAULT 0,
  "aa_count" integer DEFAULT 0,
  "ab_count" integer DEFAULT 0,
  "ac_count" integer DEFAULT 0,
  "ad_count" integer DEFAULT 0,
  "ae_count" integer DEFAULT 0,
  "af_count" integer DEFAULT 0,
  "b0_count" integer DEFAULT 0,
  "b1_count" integer DEFAULT 0,
  "b2_count" integer DEFAULT 0,
  "b3_count" integer DEFAULT 0,
  "b4_count" integer DEFAULT 0,
  "b5_count" integer DEFAULT 0,
  "b6_count" integer DEFAULT 0,
  "b7_count" integer DEFAULT 0,
  "b8_count" integer DEFAULT 0
);

-- ----------------------------
-- Records of pedes_statis
-- ----------------------------
BEGIN;
INSERT INTO "pedes_statis" VALUES (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1);
COMMIT;

-- ----------------------------
-- Auto increment value for pedes_statis
-- ----------------------------
UPDATE "main"."sqlite_sequence" SET seq = 1 WHERE name = 'pedes_statis';

PRAGMA foreign_keys = true;