"""The exported Three.js adapter needs neither React nor a JavaScript package."""

# ### Imports ###
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from housemaker.pvs_runtime import PVS_RUNTIME_JAVASCRIPT, write_pvs_runtime_module

# ### Dependency-free JavaScript scene fixtures ###
JAVASCRIPT_FIXTURES = r'''
function node(name, extra = {}) {
  return { name, visible: true, castShadow: false, frustumCulled: true,
    isMesh: true, children: [], ...extra };
}
function scene(...children) {
  const root = node("root", { isMesh: false, children });
  root.traverse = callback => {
    function visit(current) {
      callback(current);
      current.children.forEach(visit);
    }
    visit(root);
  };
  return root;
}
function manifest() {
  return { version: 1, algorithmVersion: 1, coordinateSystem: "gltf-y-up",
    sourceSignature: "a".repeat(64), glbSha256: "b".repeat(64),
    fallback: "frustum-only", unlistedNodes: "frustum-only",
    clusters: [
      { id: "a", nodeNames: ["A"], instanceIds: [], min: [-2,-2,-2], max: [0,2,2] },
      { id: "b", nodeNames: ["B"], instanceIds: [], min: [0,-2,-2], max: [2,2,2] }
    ],
    cells: [
      { id: "left", min: [-2,-2,-2], max: [0,2,2], visibleClusterIds: ["a"] },
      { id: "right", min: [0,-2,-2], max: [2,2,2], visibleClusterIds: ["b"] }
    ]
  };
}
const left = { x: -1, y: 0, z: 0 }, right = { x: 1, y: 0, z: 0 };
'''


def _run_javascript(tmp_path, body):
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("Node.js is unavailable; runtime behavior tests require Node.")
    module_path = write_pvs_runtime_module(tmp_path / "house.glb")
    program = (
        'import assert from "node:assert/strict";\n'
        f"const {{ createPvsController }} = await import({json.dumps(module_path.as_uri())});\n"
        + JAVASCRIPT_FIXTURES + "\n" + body
    )
    result = subprocess.run(
        [node_path, "--input-type=module", "-e", program],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


# ### Written runtime asset ###
def test_runtime_module_is_written_next_to_glb_without_changing_glb(tmp_path):
    glb_path = tmp_path / "my house.GLB"
    glb_path.write_bytes(b"original glb")
    module_path = write_pvs_runtime_module(glb_path)
    assert module_path.name == "my house.pvs.mjs"
    assert module_path.read_text(encoding="utf-8") == PVS_RUNTIME_JAVASCRIPT
    assert glb_path.read_bytes() == b"original glb"
    assert "useFrame" in PVS_RUNTIME_JAVASCRIPT
    assert "export function createPvsController" in PVS_RUNTIME_JAVASCRIPT


def test_runtime_writer_requires_glb_filename(tmp_path):
    with pytest.raises(ValueError, match=".glb"):
        write_pvs_runtime_module(tmp_path / "other.txt")
    assert not (tmp_path / "other.pvs.mjs").exists()


# ### Unioned cells, fallback and update caching ###
def test_runtime_unions_boundaries_and_outside_invalid_position_fail_open(tmp_path):
    _run_javascript(tmp_path, r'''
      const a = node("A"), b = node("B"), extra = node("unlisted");
      const pvs = createPvsController({ scene: scene(a,b,extra), visibility: manifest() });
      assert.equal(pvs.valid, true);
      assert.equal(pvs.update(left), true);
      assert.equal(a.visible, true); assert.equal(b.visible, false);
      assert.equal(pvs.update({x:-0.5,y:0,z:0}), false);
      pvs.update({x:0,y:0,z:0});
      assert.equal(a.visible, true); assert.equal(b.visible, true);
      pvs.update(right); assert.equal(a.visible, false);
      pvs.update({x:50,y:0,z:0});
      assert.equal(a.visible, true); assert.equal(b.visible, true);
      pvs.update(left); pvs.update({x:NaN,y:0,z:0});
      assert.equal(b.visible, true);
      assert.equal(extra.visible, true); assert.equal(extra.frustumCulled, true);
    ''')


def test_repeated_same_cell_updates_never_rewrite_visible_flags(tmp_path):
    _run_javascript(tmp_path, r'''
      let writes = 0, value = true;
      const a = node("A"), b = node("B");
      Object.defineProperty(b, "visible", {get:()=>value, set:v=>{writes++;value=v;}});
      const pvs = createPvsController({scene:scene(a,b),visibility:manifest()});
      pvs.update(left); assert.equal(writes,1);
      for(let i=0;i<100;i++) assert.equal(pvs.update(left),false);
      assert.equal(writes,1);
    ''')


# ### Manual visibility baselines and lifecycle ###
def test_original_hidden_flags_external_writes_and_manual_hidden_during_cull_survive(tmp_path):
    _run_javascript(tmp_path, r'''
      const a = node("A",{visible:false}), b = node("B");
      const pvs = createPvsController({scene:scene(a,b),visibility:manifest()});
      pvs.update(left); assert.equal(a.visible,false); assert.equal(b.visible,false);
      pvs.setNodeVisible("B",false); // Already PVS-hidden; explicit baseline setter.
      pvs.update(right); assert.equal(a.visible,false); assert.equal(b.visible,false);
      pvs.setNodeVisible("A",true); pvs.update(left); assert.equal(a.visible,true);
      a.visible = false; // Observable external user hide while PVS-visible.
      pvs.update(left); assert.equal(a.visible,false);
      pvs.dispose(); assert.equal(a.visible,false); assert.equal(b.visible,false);
      assert.equal(pvs.enabled,false); assert.equal(pvs.update(right),false);
    ''')


def test_disabling_reset_and_dispose_restore_only_visibility_not_frustum_or_collisions(tmp_path):
    _run_javascript(tmp_path, r'''
      const a = node("A"), b = node("B",{collider:{enabled:true}});
      const pvs = createPvsController({scene:scene(a,b),visibility:manifest()});
      pvs.update(left); assert.equal(b.visible,false);
      pvs.setEnabled(false); assert.equal(b.visible,true);
      pvs.update(left); assert.equal(b.visible,true);
      pvs.setEnabled(true); pvs.update(left); assert.equal(b.visible,false);
      pvs.reset(); assert.equal(b.visible,true);
      pvs.update(left); pvs.dispose(); assert.equal(b.visible,true);
      assert.equal(b.frustumCulled,true); assert.equal(b.collider.enabled,true);
    ''')


# ### Shadow casters, instancing and unmapped scene nodes ###
def test_shadow_casters_are_kept_by_default_and_explicit_opt_in_can_cull(tmp_path):
    _run_javascript(tmp_path, r'''
      const a = node("A"), b = node("B",{castShadow:true});
      const root = scene(a,b);
      const safe = createPvsController({scene:root,visibility:manifest()});
      safe.update(left); assert.equal(b.visible,true); safe.dispose();
      const optin = createPvsController({scene:root,visibility:manifest(),cullShadowCasters:true});
      optin.update(left); assert.equal(b.visible,false); optin.dispose();
      assert.equal(b.castShadow,true);
    ''')


def test_shadow_flag_changes_restore_hidden_node_even_with_same_cell_membership(tmp_path):
    _run_javascript(tmp_path, r'''
      const a=node("A"), b=node("B");
      const pvs=createPvsController({scene:scene(a,b),visibility:manifest()});
      pvs.update(left); assert.equal(b.visible,false);
      b.castShadow=true;
      pvs.update(left); assert.equal(b.visible,true);
      b.castShadow=false;
      pvs.update(left); assert.equal(b.visible,false);
    ''')


def test_user_setter_never_resurrects_other_externally_hidden_node(tmp_path):
    _run_javascript(tmp_path, r'''
      const a=node("A"), b=node("B");
      const pvs=createPvsController({scene:scene(a,b),visibility:manifest()});
      pvs.update({x:0,y:0,z:0});
      a.visible=false;
      pvs.setNodeVisible("B",false);
      assert.equal(a.visible,false); assert.equal(b.visible,false);
      pvs.dispose(); assert.equal(a.visible,false); assert.equal(b.visible,false);
    ''')


def test_mapped_parent_never_hides_unmapped_or_other_cluster_child(tmp_path):
    _run_javascript(tmp_path, r'''
      const a = node("A"), dynamic = node("dynamic"), b = node("B",{children:[dynamic]});
      const pvs = createPvsController({scene:scene(a,b),visibility:manifest()});
      pvs.update(left); assert.equal(b.visible,true); assert.equal(dynamic.visible,true);
      const pvs2 = createPvsController({scene:scene(node("B",{children:[node("A")]})),visibility:manifest()});
      pvs2.update(left);
    ''')


def test_instance_callback_is_optional_and_never_hides_instanced_mesh_group(tmp_path):
    _run_javascript(tmp_path, r'''
      const a = node("A"), b = node("B",{isInstancedMesh:true});
      const data = manifest(); data.clusters[1].instanceIds=["instance-1"];
      const root=scene(a,b);
      const noCallback=createPvsController({scene:root,visibility:data});
      noCallback.update(left); assert.equal(b.visible,true); noCallback.dispose();
      const calls=[];
      const pvs=createPvsController({scene:root,visibility:data,
        instanceVisibility:(id,visible)=>calls.push([id,visible])});
      pvs.update(left); assert.deepEqual(calls,[["instance-1",false]]);
      assert.equal(b.visible,true);
      pvs.update(left); assert.equal(calls.length,1);
      pvs.dispose(); assert.deepEqual(calls[1],["instance-1",true]);
    ''')


def test_instance_shadow_caster_keeps_instance_callback_visible_by_default(tmp_path):
    _run_javascript(tmp_path, r'''
      const data=manifest(); data.clusters[1].instanceIds=["instance-1"];
      const calls=[];
      const pvs=createPvsController({scene:scene(node("A"),node("B",{
        isInstancedMesh:true,castShadow:true})),visibility:data,
        instanceVisibility:(id,visible)=>calls.push([id,visible])});
      pvs.update(left); assert.deepEqual(calls,[]);
    ''')


def test_instance_user_baselines_persist_after_pvs_reset_and_disposal(tmp_path):
    _run_javascript(tmp_path, r'''
      const data=manifest(); data.clusters[1].instanceIds=["instance-1","instance-2"];
      const state=new Map([["instance-1",false],["instance-2",true]]);
      const pvs=createPvsController({scene:scene(node("A"),node("B",{isInstancedMesh:true})),
        visibility:data,getInstanceVisible:id=>state.get(id),
        setInstanceVisible:(id,visible)=>state.set(id,visible)});
      pvs.update(left);
      assert.equal(state.get("instance-1"),false); assert.equal(state.get("instance-2"),false);
      pvs.setInstanceVisible("instance-2",false);
      pvs.update(right); assert.equal(state.get("instance-2"),false);
      pvs.dispose();
      assert.equal(state.get("instance-1"),false); assert.equal(state.get("instance-2"),false);
    ''')


# ### Schema, stale payload and ambiguous mapping fail-open guards ###
@pytest.mark.parametrize("mutation", (
    "data.version=2", "data.coordinateSystem='canonical-z-up'",
    "data.glbSha256='not-a-hash'", "data.cells[0].min=[0,0,0]",
    "data.cells[0].visibleClusterIds=['unknown']",
    "data.clusters[1].nodeNames=['A']", "data.cells=[]",
))
def test_malformed_manifest_never_changes_scene_visibility(tmp_path, mutation):
    _run_javascript(tmp_path, r'''
      const a=node("A"), b=node("B"), data=manifest();
    ''' + mutation + r''';
      const pvs=createPvsController({scene:scene(a,b),visibility:data});
      assert.equal(pvs.valid,false); assert.equal(pvs.update(left),false);
      assert.equal(a.visible,true); assert.equal(b.visible,true);
    ''')


def test_loaded_glb_hash_mismatch_and_ambiguous_node_names_stay_visible(tmp_path):
    _run_javascript(tmp_path, r'''
      const a=node("A"), b=node("B");
      const stale=createPvsController({scene:scene(a,b),visibility:manifest(),
        expectedGlbSha256:"c".repeat(64)});
      assert.equal(stale.valid,false); stale.update(left); assert.equal(b.visible,true);
      const matching=createPvsController({scene:scene(a,b),visibility:manifest(),
        expectedGlbSha256:"B".repeat(64)});
      assert.equal(matching.valid,true); matching.dispose();
      const b2=node("B");
      const ambiguous=createPvsController({scene:scene(a,b,b2),visibility:manifest()});
      ambiguous.update(left); assert.equal(b.visible,true); assert.equal(b2.visible,true);
    ''')
