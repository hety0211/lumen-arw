"""1.4.0: visible-area rendering must match the whole-frame render it replaces."""
import copy
import cv2
import numpy as np
import pytest
from lumen import engine, model, retouch, selection, viewport


def frame(h=430, w=610, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([xx / w, yy / h, .5 + .4 * np.sin(xx / 23) * np.cos(yy / 17)], axis=2)
    texture = rng.normal(0, .04, (h, w, 3)).astype(np.float32)
    edges = ((xx // 37 + yy // 29) % 2)[..., None] * .25
    return np.clip(.6 * base + texture + edges, 0, 1).astype(np.float32) ** 2.2


RECTS = [(0, 0, 610, 430), (0, 0, 128, 96), (201, 133, 377, 301), (480, 300, 610, 430), (5, 250, 600, 262)]


def recipes():
    plain = model.recipe()
    plain['adjustments'].update(exposure=.4, contrast=12, shadows=20, highlights=-30, saturation=15, vibrance=10)
    plain['hsl'][2] = [10, -20, 5]
    plain['curves']['RGB'] = [[0., 0.], [.3, .25], [1., 1.]]
    plain['curve_mode'] = 'smooth'
    plain['grading']['shadows'] = [210, 30]
    plain['wb_gain'] = [1.05, 1., .93]
    plain['develop'] = dict(mode='camera', curve=[[0., 0.], [.1, .2], [1., 1.]], source='test')
    spatial = copy.deepcopy(plain)
    spatial['adjustments'].update(denoise=30, color_noise=40, clarity=25, texture=-20, sharpness=60)
    hazy = copy.deepcopy(spatial)
    hazy['adjustments'].update(dehaze=45)
    clearer = copy.deepcopy(plain)
    clearer['adjustments'].update(dehaze=-30)
    masked = copy.deepcopy(spatial)
    brush = model.new_mask('brush', 1)
    brush['strokes'] = [dict(points=[[.1, .2], [.5, .45], [.8, .3]], radius=.05, erase=False),
                        dict(points=[[.45, .4]], radius=.02, erase=True)]
    brush['adjustments'].update(exposure=.5, clarity=40, sharpness=30)
    linear = model.new_mask('linear', 2)
    linear.update(start=[.2, .1], end=[.7, .9])
    linear['adjustments'].update(saturation=-40, temperature=20)
    radial = model.new_mask('radial', 3)
    radial.update(start=[.3, .3], end=[.9, .8], feather=40, invert=True)
    radial['adjustments'].update(shadows=30)
    luminance = model.new_mask('luminance', 4)
    luminance['adjustments'].update(highlights=-40)
    sky = model.new_mask('sky', 5)
    small = cv2.GaussianBlur((np.random.default_rng(5).random((60, 90)) > .5).astype(np.float32), (0, 0), 3)
    sky.update(raster=selection.encode(small), feather=10.,
               strokes=[dict(points=[[.6, .6], [.7, .65]], radius=.03, erase=True)])
    sky['adjustments'].update(exposure=-.3, clarity=20)
    masked['masks'] = [brush, linear, radial, luminance, sky]
    repaired = copy.deepcopy(plain)
    repaired['retouch'] = [
        dict(kind='heal', points=[[.33, .4], [.36, .42]], radius=.02, enabled=True, feather=50, opacity=100),
        dict(kind='clone', points=[[.52, .55], [.56, .6]], radius=.03, offset=[-.3, -.2], enabled=True, feather=30, opacity=90),
        dict(kind='clone', points=[[.2, .5]], radius=.025, offset=[.4, .1], enabled=True, feather=0, opacity=100),
        dict(kind='heal', points=[[.9, .9]], radius=.02, enabled=False, feather=50, opacity=100)]
    finished = copy.deepcopy(spatial)
    finished['effects'].update(vignette=-40, grain=30, grain_size=45)
    finished['crop'] = [.1, .05, .85, .9]
    return dict(plain=plain, spatial=spatial, hazy=hazy, clearer=clearer, masked=masked,
                repaired=repaired, finished=finished)


@pytest.mark.parametrize('size,shape', [((90, 60), (6336, 9504)), ((1626, 1083), (6336, 9504)),
                                        ((47, 33), (5464, 8192)), ((1365, 2048), (1067, 1600))])
def test_resize_region_matches_opencv(size, shape):
    small = np.random.default_rng(1).random(size[::-1]).astype(np.float32)
    full = cv2.resize(small, shape[::-1], interpolation=cv2.INTER_LINEAR)
    h, w = shape
    for x0, y0, x1, y1 in [(0, 0, w, h), (w // 3, h // 4, w // 3 + 777, h // 4 + 555), (w - 300, h - 200, w, h)]:
        block = engine.resize_region(small, (w, h), engine.Area(w, h, x0, y0, x1, y1))
        np.testing.assert_allclose(block, full[y0:y1, x0:x1], atol=1e-6)


def test_retouch_block_reads_clone_sources_outside_it():
    source = frame()
    ops = recipes()['repaired']['retouch']
    whole = retouch.apply(source, ops)
    for x0, y0, x1, y1 in RECTS + [(300, 220, 360, 280), (100, 150, 160, 200)]:
        block = retouch.apply(source, ops, (x0, y0, x1, y1))
        np.testing.assert_array_equal(block, whole[y0:y1, x0:x1])


@pytest.mark.parametrize('name', ['plain', 'spatial', 'hazy', 'clearer', 'masked', 'repaired', 'finished'])
def test_region_render_matches_whole_frame(name):
    source, edits = frame(), recipes()[name]
    whole = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False, detail_scale=.7)
    for x0, y0, x1, y1 in RECTS:
        block = engine.process_region(source, edits, engine.Backend('cpu'), (x0, y0, x1, y1), detail_scale=.7)
        assert block.shape == (y1 - y0, x1 - x0, 3)
        difference = np.abs(block - whole[y0:y1, x0:x1])
        # Bilateral filters bin their range weights by the block's value range: ~1e-5 at most.
        assert difference.max() < 2e-4, (name, (x0, y0), float(difference.max()))
        assert difference.mean() < 1e-5


def test_region_dehaze_uses_supplied_frame_statistics():
    source, edits = frame(), recipes()['hazy']
    preview = engine.resize_limit(source, 300)
    context = engine.dehaze_context_for(preview, edits, engine.Backend('cpu'), .5)
    block = engine.process_region(source, edits, engine.Backend('cpu'), (200, 100, 400, 300), dehaze_context=context)
    own = engine.process_region(source, edits, engine.Backend('cpu'), (200, 100, 400, 300))
    assert np.abs(block - own).max() > 1e-4 and np.isfinite(block).all()


@pytest.mark.parametrize('rotation', [0, 1, 2, 3])
@pytest.mark.parametrize('straighten', [0., 4.5, -11.])
def test_display_block_matches_crop_rotate(rotation, straighten):
    source, edits = frame(), recipes()['finished']
    edits.update(rotation=rotation, straighten=straighten)
    whole = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False)
    display = engine.crop_rotate(whole, edits)
    h, w = display.shape[:2]
    assert engine.display_size(edits, (source.shape[1], source.shape[0])) == (w, h)
    for x0, y0, x1, y1 in [(0, 0, w, h), (0, 0, w // 3, h // 2), (w // 4, h // 3, w - 7, h - 5), (w - 50, h - 60, w, h)]:
        block = engine.render_display(source, edits, engine.Backend('cpu'), (x0, y0, x1, y1), final=True)
        assert block.shape == (y1 - y0, x1 - x0, 3)
        difference = np.abs(block - display[y0:y1, x0:x1])
        if straighten:
            # Bicubic positions are rounded to 1/32 pixel per block origin: invisible, not bit-exact.
            assert difference.mean() < 2e-3 and np.percentile(difference, 99.9) < .03, float(difference.max())
        else:
            assert difference.max() < 2e-4
    original = engine.render_display(source, edits, None, (3, 4, w - 3, h - 4), final=True, kind='original')
    expected = engine.crop_rotate(np.clip(engine.to_srgb(engine.develop.apply(source, edits['develop'])), 0, 1), edits)
    assert np.abs(original - expected[4:h - 4, 3:w - 3]).mean() < 2e-3


def test_masks_cover_blocks_like_the_whole_frame():
    edits = recipes()['masked']
    shape = (430, 610)
    reference = np.clip(engine.to_srgb(frame()), 0, 1)
    for mask in edits['masks']:
        whole = engine.mask_alpha(mask, shape, reference)
        for x0, y0, x1, y1 in RECTS:
            area = engine.Area(610, 430, x0, y0, x1, y1)
            block = engine.mask_alpha(mask, shape, reference[y0:y1, x0:x1], area)
            np.testing.assert_allclose(block, whole[y0:y1, x0:x1], atol=2e-6, err_msg=mask['kind'])


def test_levels_and_visible_tiles():
    assert viewport.level_for(1.5) == 0 and viewport.level_for(1) == 0 and viewport.level_for(.6) == 0
    assert viewport.level_for(.5) == 1 and viewport.level_for(.26) == 1 and viewport.level_for(.2) == 2
    assert viewport.level_size(9504, 6336, 2) == (2376, 1584)
    # 100 % view of a 9504 px frame in a 1000 × 700 widget, centred.
    tiles = viewport.visible_tiles((-4252, -2818, 9504, 6336), (0, 0, 1000, 700), (9504, 6336))
    assert {t[0] for t in tiles} == {8, 9, 10} and {t[1] for t in tiles} == {5, 6}
    assert viewport.visible_tiles((2000, 0, 100, 100), (0, 0, 1000, 700), (9504, 6336)) == []


def test_group_covers_tiles_with_few_rectangles():
    full = [(x, y) for y in range(3) for x in range(4)]
    assert viewport.group(full) == [(0, 0, 4, 3)]
    strip = [(5, y) for y in range(4)] + [(x, 4) for x in range(6)]
    rects = viewport.group(strip)
    covered = {(x, y) for x0, y0, x1, y1 in rects for y in range(y0, y1) for x in range(x0, x1)}
    assert covered == set(strip) and len(rects) == 2
    assert all((x1 - x0) * (y1 - y0) <= 4 for x0, y0, x1, y1 in viewport.group(full, 4))


def test_tile_store_evicts_least_recent_and_drops_stale_layers():
    store = viewport.TileStore(max_bytes=3)
    tile = lambda: type('T', (), {'nbytes': 1})()
    store.add('edited', 1, 0, {(0, 0): tile(), (1, 0): tile()})
    store.missing('edited', 1, 0, [(0, 0)])
    store.add('edited', 1, 0, {(2, 0): tile(), (3, 0): tile()})
    assert set(store.layer('edited', 1, 0)) == {(0, 0), (2, 0), (3, 0)} and store.bytes == 3
    store.add('original', 7, 0, {(0, 0): tile()})
    store.retain({('original', 7)})
    assert not store.layer('edited', 1, 0) and store.bytes == 1


def test_tiles_convert_to_screen_images_off_the_gui_thread():
    rgb = np.zeros((40, 50, 3), np.float32)
    rgb[:10] = 1
    rgb[10:20, :, 0] = .5
    tile = viewport.make_tile(rgb, 512, 0)
    assert (tile.image.width(), tile.image.height()) == (50, 40)
    assert tile.image.pixelColor(3, 15).red() == 127 and tile.image.pixelColor(3, 15).green() == 0
    assert tile.highlights is not None and tile.shadows is not None
    assert tile.highlights.pixelColor(0, 0).alpha() == 210 and tile.highlights.pixelColor(0, 30).alpha() == 0
