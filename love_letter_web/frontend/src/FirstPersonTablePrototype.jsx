import React from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const GOTHIC_TABLE_URL = "/models/gothic_coffee_table/gothic_coffee_table_1k.gltf";

function CameraRig() {
  const { camera } = useThree();
  const lookTarget = React.useMemo(() => ({ x: 0, y: 0.7, z: 0.42 }), []);

  React.useEffect(() => {
    camera.position.set(0, 1.85, 2.78);
    camera.rotation.set(0, 0, 0);
    camera.lookAt(lookTarget.x, lookTarget.y, lookTarget.z);
  }, [camera, lookTarget]);

  useFrame(({ clock }) => {
    const breath = Math.sin(clock.elapsedTime * 0.72) * 0.012;
    camera.position.y = 1.85 + breath;
    camera.lookAt(lookTarget.x, lookTarget.y + breath * 0.35, lookTarget.z);
  });

  return null;
}

function Box({ position, scale, color, roughness = 0.82 }) {
  return (
    <mesh position={position} scale={scale} castShadow receiveShadow>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={color} roughness={roughness} />
    </mesh>
  );
}

function GothicCoffeeTable() {
  const gltf = useLoader(GLTFLoader, GOTHIC_TABLE_URL);

  React.useEffect(() => {
    gltf.scene.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
  }, [gltf.scene]);

  return (
    <primitive
      object={gltf.scene}
      position={[0, 0.04, 0.5]}
      rotation={[0, Math.PI / 4, 0]}
      scale={[2.25, 1.58, 1.58]}
    />
  );
}

function TableScene() {
  const cardPositions = [-0.5, 0, 0.5];

  return (
    <>
      <CameraRig />
      <color attach="background" args={["#11101a"]} />
      <fog attach="fog" args={["#11101a", 4.2, 8.8]} />

      <ambientLight intensity={0.55} />
      <directionalLight
        castShadow
        position={[1.6, 3.8, 2.1]}
        intensity={1.6}
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[0, 1.8, 0.25]} intensity={0.85} color="#e8bc58" />

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, -0.1]} receiveShadow>
        <planeGeometry args={[9, 9]} />
        <meshStandardMaterial color="#21141a" roughness={0.94} />
      </mesh>

      <Box position={[0, 0.04, 1.58]} scale={[1.42, 0.08, 0.7]} color="#3a2220" />
      <Box position={[-0.98, 0.54, 1.42]} scale={[0.22, 0.78, 0.92]} color="#2a1717" />
      <Box position={[0.98, 0.54, 1.42]} scale={[0.22, 0.78, 0.92]} color="#2a1717" />

      <React.Suspense fallback={<Box position={[0, 0.72, 0.5]} scale={[3.1, 0.18, 1.8]} color="#70401d" />}>
        <GothicCoffeeTable />
      </React.Suspense>

      {cardPositions.map((x, index) => (
        <mesh
          key={x}
          position={[x, 0.945 + index * 0.003, 0.68]}
          rotation={[-Math.PI / 2, 0, (index - 1) * 0.08]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[0.32, 0.48, 0.018]} />
          <meshStandardMaterial color={index === 1 ? "#f0dfae" : "#f2c55a"} roughness={0.65} />
        </mesh>
      ))}

      <Box position={[0, 1.16, -1.55]} scale={[4.8, 2.3, 0.12]} color="#241821" />
      <Box position={[-2.25, 1.18, -1.1]} scale={[0.22, 2.15, 0.2]} color="#4a2a1d" />
      <Box position={[2.25, 1.18, -1.1]} scale={[0.22, 2.15, 0.2]} color="#4a2a1d" />
      <Box position={[0, 2.16, -1.18]} scale={[4.65, 0.2, 0.18]} color="#5f371f" />

      <mesh position={[0, 1.82, -1.42]} rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[0.46, 0.46, 0.04]} />
        <meshStandardMaterial color="#c5953c" emissive="#3a210c" roughness={0.55} />
      </mesh>
    </>
  );
}

export function FirstPersonTablePrototype() {
  return (
    <main className="first-person-prototype-screen">
      <Canvas
        shadows
        camera={{ fov: 54, near: 0.1, far: 20 }}
        dpr={[1, 1.75]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
      >
        <TableScene />
      </Canvas>
      <section className="first-person-prototype-hud" aria-label="Prototype 3D">
        <span>Prototype 3D</span>
        <strong>Vue assise à la table</strong>
        <p>Caméra fixe, table simple, cartes placeholders. Prochaine étape: vraie table, Sultan GLB, puis animations de cartes.</p>
      </section>
    </main>
  );
}

export default FirstPersonTablePrototype;
