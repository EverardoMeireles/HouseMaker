"""Export a dependency-free Three.js/r3f baked-visibility runtime adapter."""

# ### Imports ###
from __future__ import annotations

from pathlib import Path

# ### Packaged JavaScript runtime ###
PVS_RUNTIME_JAVASCRIPT = r'''/**
 * HouseMaker conservative PVS adapter. No React/Three imports are required.
 *
 * r3f integration (create/dispose once per loaded asset, not on every render):
 *   const pvs = createPvsController({ scene: gltf.scene,
 *     visibility: manifest.visibility, expectedGlbSha256: actualGlbSha256 });
 *   useFrame(({ camera }) => pvs.update(camera.getWorldPosition(scratch)));
 *   // On unmount: pvs.dispose();
 *
 * Positions must be physical glTF Y-up positions in the exported asset frame.
 * For a transformed asset root, convert the world camera position into that
 * asset frame first. The optional expected hash must come from the loaded GLB
 * payload: this module validates/compares hashes but does not fetch/hash files.
 *
 * Manual visibility controls should call setNodeVisible/setInstanceVisible.
 * External node.visible writes are detected when observable, but setting false
 * on an already PVS-hidden node is inherently indistinguishable without those
 * setters. Initially hidden nodes and the latest user baseline are preserved.
 * reset/dispose restore that baseline, not unconditional visible=true.
 *
 * Shadow casters are not culled by default. Opt in with cullShadowCasters:true
 * only if the application accepts changing offscreen shadows. Instance IDs use
 * an optional instanceVisibility(id, visible) / setInstanceVisible callback;
 * without one their render groups are never hidden. If any InstancedMesh casts
 * shadows, instance culling also stays off unless explicitly opted in.
 *
 * This adapter never changes frustumCulled, collisions, geometry or transforms.
 * Recreate it after replacing scene nodes or changing the exported hierarchy.
 */

// ### Manifest validation ###
const SHA256 = /^[0-9a-f]{64}$/i;
const MAX_CELLS = 250000;
const MAX_CLUSTERS = 100000;
const MAX_REFERENCES = 10000000;

function identifier(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 1024
    && value.trim() === value && !/[\x00-\x1f]/.test(value);
}

function identifiers(values) {
  return Array.isArray(values) && values.every(identifier)
    && new Set(values).size === values.length;
}

function bounds(value, volume = false) {
  return value && Array.isArray(value.min) && Array.isArray(value.max)
    && value.min.length === 3 && value.max.length === 3
    && value.min.every(Number.isFinite) && value.max.every(Number.isFinite)
    && value.min.every((low, axis) => volume
      ? low < value.max[axis] : low <= value.max[axis]);
}

function validatedVisibility(value, expectedHash) {
  if (!value || value.version !== 1 || value.algorithmVersion !== 1
      || value.coordinateSystem !== "gltf-y-up"
      || !SHA256.test(value.sourceSignature) || !SHA256.test(value.glbSha256)
      || value.fallback !== "frustum-only" || value.unlistedNodes !== "frustum-only"
      || (expectedHash !== undefined && (!SHA256.test(expectedHash)
        || expectedHash.toLowerCase() !== value.glbSha256.toLowerCase()))
      || !Array.isArray(value.clusters) || value.clusters.length === 0
      || value.clusters.length > MAX_CLUSTERS
      || !Array.isArray(value.cells) || value.cells.length === 0
      || value.cells.length > MAX_CELLS) return null;
  const clusterIds = new Set(), nodeNames = new Set(), instanceIds = new Set();
  const clusters = [];
  for (const cluster of value.clusters) {
    const instances = cluster && (cluster.instanceIds ?? []);
    if (!cluster || !identifier(cluster.id) || clusterIds.has(cluster.id)
        || !identifiers(cluster.nodeNames) || !identifiers(instances)
        || (!cluster.nodeNames.length && !instances.length) || !bounds(cluster)) return null;
    if (cluster.nodeNames.some(name => nodeNames.has(name))
        || instances.some(id => instanceIds.has(id))) return null;
    clusterIds.add(cluster.id);
    cluster.nodeNames.forEach(name => nodeNames.add(name));
    instances.forEach(id => instanceIds.add(id));
    clusters.push({ id: cluster.id, nodeNames: [...cluster.nodeNames], instanceIds: [...instances] });
  }
  const cellIds = new Set(), cells = [];
  let references = 0;
  for (const cell of value.cells) {
    if (!cell || !identifier(cell.id) || cellIds.has(cell.id) || !bounds(cell, true)
        || !identifiers(cell.visibleClusterIds)
        || cell.visibleClusterIds.some(id => !clusterIds.has(id))) return null;
    references += cell.visibleClusterIds.length;
    if (references > MAX_REFERENCES) return null;
    cellIds.add(cell.id);
    cells.push({ id: cell.id, min: [...cell.min], max: [...cell.max],
      visible: [...cell.visibleClusterIds] });
  }
  return { clusters, cells };
}

// ### Scene traversal and conservative node ownership ###
function visit(root, callback) {
  if (root && typeof root.traverse === "function") root.traverse(callback);
  else {
    callback(root);
    for (const child of root?.children ?? []) visit(child, callback);
  }
}

function hasProperty(root, property) {
  let found = false;
  visit(root, node => { if (node && node[property]) found = true; });
  return found;
}

function protectedSubtree(node, clusterId, owners, cullShadowCasters) {
  let protectedNode = false;
  visit(node, child => {
    if (!child) return;
    if (child.isInstancedMesh || (!cullShadowCasters && child.castShadow)) protectedNode = true;
    // Never hide unlisted descendants or nodes belonging to another cluster.
    if (child !== node && owners.get(child) !== clusterId) protectedNode = true;
  });
  return protectedNode;
}

// ### Controller creation and captured visibility baselines ###
export function createPvsController(options = {}) {
  const { scene, visibility, expectedGlbSha256, cullShadowCasters = false } = options;
  const data = validatedVisibility(visibility, expectedGlbSha256);
  const instanceWriter = options.setInstanceVisible ?? options.instanceVisibility;
  const instanceReader = options.getInstanceVisible;
  let enabled = true, disposed = false, membershipKey = null, allowed = null;
  let reason = data ? "" : "Invalid or mismatched PVS manifest; frustum-only fallback.";
  const nodes = [], instances = [], nodesByName = new Map(), instancesById = new Map();
  const sceneNames = new Map(), owners = new Map();
  const instanceRoots = [];
  let instanceShadowCaster = false;
  if (data && scene) {
    visit(scene, node => {
      if (!node) return;
      if (identifier(node.name)) {
        if (!sceneNames.has(node.name)) sceneNames.set(node.name, []);
        sceneNames.get(node.name).push(node);
      }
      if (node.isInstancedMesh) {
        instanceRoots.push(node);
        if (hasProperty(node, "castShadow")) instanceShadowCaster = true;
      }
    });
    for (const cluster of data.clusters) {
      for (const name of cluster.nodeNames) {
        const matches = sceneNames.get(name) ?? [];
        // Missing or ambiguous names are unmapped: leave them entirely alone.
        if (matches.length !== 1) continue;
        const node = matches[0];
        owners.set(node, cluster.id);
        const record = { name, node, clusterId: cluster.id,
          baseline: node.visible !== false, lastWritten: node.visible !== false };
        nodes.push(record);
        nodesByName.set(name, record);
      }
      if (typeof instanceWriter === "function") for (const id of cluster.instanceIds) {
        const baseline = typeof instanceReader === "function" ? instanceReader(id) !== false : true;
        const record = { id, clusterId: cluster.id, baseline, lastWritten: baseline };
        instances.push(record);
        instancesById.set(id, record);
      }
    }
  } else if (data) reason = "Missing scene; frustum-only fallback.";
  const valid = Boolean(data && scene);

  // ### Visibility updates and manual state reconciliation ###
  function captureExternalWrites() {
    let changed = false;
    for (const record of nodes) {
      const visible = record.node.visible !== false;
      if (visible !== record.lastWritten) {
        record.baseline = visible;
        record.lastWritten = visible;
        changed = true;
      }
      const protectedNode = protectedSubtree(
        record.node, record.clusterId, owners, cullShadowCasters);
      if (record.protectedNode !== protectedNode) {
        record.protectedNode = protectedNode;
        changed = true;
      }
    }
    const currentInstanceShadowCaster = instanceRoots.some(root => hasProperty(root, "castShadow"));
    if (currentInstanceShadowCaster !== instanceShadowCaster) {
      instanceShadowCaster = currentInstanceShadowCaster;
      changed = true;
    }
    if (typeof instanceReader === "function") for (const record of instances) {
      const visible = instanceReader(record.id) !== false;
      if (visible !== record.lastWritten) {
        record.baseline = visible;
        record.lastWritten = visible;
        changed = true;
      }
    }
    return changed;
  }

  function applyVisibility(visibleClusters) {
    let changed = false;
    for (const record of nodes) {
      const protectedNode = protectedSubtree(record.node, record.clusterId, owners, cullShadowCasters);
      record.protectedNode = protectedNode;
      const visible = record.baseline && (protectedNode || visibleClusters === null
        || visibleClusters.has(record.clusterId));
      if ((record.node.visible !== false) !== visible) {
        record.node.visible = visible;
        changed = true;
      }
      record.lastWritten = visible;
    }
    for (const record of instances) {
      const visible = record.baseline && ((!cullShadowCasters && instanceShadowCaster)
        || visibleClusters === null || visibleClusters.has(record.clusterId));
      if (record.lastWritten !== visible) {
        instanceWriter(record.id, visible);
        changed = true;
      }
      record.lastWritten = visible;
    }
    return changed;
  }

  function reset() {
    if (disposed) return false;
    captureExternalWrites();
    membershipKey = null;
    allowed = null;
    return applyVisibility(null);
  }

  function update(position) {
    if (disposed || !valid || !enabled) return false;
    const externallyChanged = captureExternalWrites();
    const finite = position && [position.x, position.y, position.z].every(Number.isFinite);
    const matches = [], visible = new Set();
    const point = finite ? [position.x, position.y, position.z] : [];
    if (finite) for (let index = 0; index < data.cells.length; index += 1) {
      const cell = data.cells[index];
      if (point.every((value, axis) => cell.min[axis] <= value && value <= cell.max[axis])) {
        matches.push(index);
        for (const id of cell.visible) visible.add(id);
      }
    }
    // Empty/invalid/outside coverage is intentionally all-visible.
    const key = matches.length ? matches.join(",") : "outside";
    if (!externallyChanged && key === membershipKey) return false;
    membershipKey = key;
    allowed = matches.length ? visible : null;
    return applyVisibility(allowed);
  }

  // ### Public lifecycle and user visibility API ###
  return {
    valid,
    reason,
    update,
    reset,
    get enabled() { return enabled && !disposed; },
    setEnabled(value) {
      if (disposed) return false;
      const next = Boolean(value);
      if (next === enabled) return false;
      enabled = next;
      return reset();
    },
    setNodeVisible(name, visible) {
      if (disposed) return false;
      const record = nodesByName.get(name);
      if (!record) return false;
      captureExternalWrites();
      record.baseline = Boolean(visible);
      return applyVisibility(enabled ? allowed : null);
    },
    setInstanceVisible(id, visible) {
      if (disposed) return false;
      const record = instancesById.get(id);
      if (!record) return false;
      captureExternalWrites();
      record.baseline = Boolean(visible);
      return applyVisibility(enabled ? allowed : null);
    },
    dispose() {
      if (disposed) return false;
      const changed = reset();
      disposed = true;
      return changed;
    },
  };
}
'''


# ### Exported runtime module asset ###
def write_pvs_runtime_module(glb_path: str | Path) -> Path:
    """Write ``<GLB stem>.pvs.mjs`` for the separately integrated web viewer."""

    source = Path(glb_path)
    if source.suffix.lower() != ".glb":
        raise ValueError("The PVS runtime module requires an exported .glb path.")
    destination = source.with_name(f"{source.stem}.pvs.mjs")
    destination.write_text(PVS_RUNTIME_JAVASCRIPT, encoding="utf-8")
    return destination
