# Release-gate inventory

This is a factual pre-release inventory for the unpublished 0.1.0 candidate.
It is not a legal opinion or a rights clearance. Package allowlists exclude
all demo/assets media and all model/cache payloads.

## Human gates

| Gate | State | Required action |
| --- | --- | --- |
| CRIT-0001 goat fixture rights/history | ACCEPTED / CLEARED | Repository owner confirmed redistribution rights; no history rewrite or visibility change is required. Current exclusions remain defense in depth. |
| Model deployment/commercial rights | OPEN / BLOCKING | Human/legal review of SAM2, CLIP and BLIP3 cards/licenses; weights remain operator assets. |
| Tracked media rights | REVIEW REQUIRED | Confirm rights for historical repository media; no absence-of-evidence inference. |
| GitHub security settings | RECOMMENDATION | Owner should review Dependabot security updates, secret scanning/push protection and branch protection; current settings are not called enabled by this repository. |
| Gateway/container/public exposure | NOT INCLUDED | Separate repository/order and separate topology/auth review required. |

## Reviewed repository media inventory

The following paths were tracked at the start of release review. Every listed
media payload remains excluded from the wheel and sdist unless an explicit
release allowlist later changes that policy.

### Repository assets

assets/banner.jpg, assets/banner.png, assets/icon-small.jpg,
assets/icon-small.png, assets/icon.jpg, assets/icon.png, assets/logo.jpg,
assets/logo.png.

### Demo images

demos/01-peppers.jpg, demos/02-peppers.jpg, demos/03-peppers.jpg,
demos/04-harbor.jpg, demos/05-harbor.jpg, demos/06-harbor.jpg,
demos/millet/millet1.jpg, demos/millet/millet2.jpg,
demos/overwater/davimar_seq_15_00305.jpg,
demos/overwater/inhouse_seq_01_00120.jpg,
demos/overwater/inhouse_seq_23_00060.jpg,
demos/overwater/inhouse_seq_28_00030.jpg,
demos/overwater/inhouse_seq_410_00075.jpg,
demos/overwater/inhouse_seq_44_00430.jpg,
demos/overwater/mastr_0095_00009.jpg,
demos/overwater/mastr_0264_00009.jpg,
demos/overwater/mastr_0483_00009.jpg,
demos/overwater/yt019_03_00263.jpg, demos/overwater/yt021_02_00149.jpg,
demos/overwater/yt039_03_00240.jpg,
demos/tomato/2022-07-22-16-25-44-48.jpg,
demos/tomato/2022-07-22-17-35-54-17.jpg,
demos/tomato/2022-07-22-22-15-39-09.jpg,
demos/tomato/2022-07-22-22-16-48-14.jpg,
demos/tomato/2022-07-22-22-33-15-01.jpg,
demos/tomato/2022-07-22-22-51-46-72.jpg,
demos/tomato/picture-av-2023-07-04-10-46-31-207301.jpg,
demos/tomato/picture-av-2023-07-04-11-00-44-327878.jpg,
demos/tomato/picture-av-2023-11-21-13-49-10-515896.jpg,
demos/underwater/Blue Tang_20180819_160313A_033.jpg,
demos/underwater/Blue Tang_20180819_160313A_035.jpg,
demos/underwater/Manta ray_20170106_032314A_001.jpg,
demos/underwater/Manta ray_20170106_032607A_024.jpg,
demos/underwater/Manta ray_20220816_104201A_009.jpg,
demos/underwater/Powder Blue Tang_20170105_051822A_001.jpg,
demos/underwater/Powder Blue Tang_20170105_051950A_005.jpg,
demos/underwater/Parrot Fish_20170106_023727A_606.jpg,
demos/underwater/Turtle_20230815_095315A_029.jpg.

The `demos/glasswool/`, `demos/icecream/`, `demos/industrial/`, and
`demos/soccer/` datasets were intentionally removed from every writable Git ref
through a coordinated history rewrite. Their similarly named YAML examples in
`configs/` remain because they contain configuration text, not the removed
images.

The four goat paths are `configs/goats.yaml`, `configs/goats2.yaml`,
`demos/goats/goats1.jpg`, and `demos/goats/goats2.jpg`. Their redistribution
rights are confirmed and CRIT-0001 is accepted. They remain ignored,
operator-supplied academic inputs and excluded from build artifacts as a
deliberate defense-in-depth distribution policy.

## Artifact and model policy

Wheels/sdists are source allowlists verified by
scripts/verify_release_artifacts.py; all media, demos, caches, weights,
outputs, OAP transcript material, credentials and private environment files
are denied. SAM2 code and model assets are revision-pinned; its weights
are downloaded only by the operator. CLIP's pinned model card has no SPDX
deployment license. BLIP3/XGen-MM is exercised only through the 007-b low-card
sequential qualification and remains CC-BY-NC/research-only. Commercial or deployed model use and any weight
redistribution require human/legal clearance.
