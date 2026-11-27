const folder=new URL('./vendor/ffmpeg/',import.meta.url);

export async function convertVideo(file,signal,progress) {
  if (file.size>250*1024**2) throw new Error('Para este archivo grande, conviértelo a MP4 H.264 o WebM antes de abrirlo.');
  const {FFmpeg}=await import(new URL('index.js',folder).href);
  const converter=new FFmpeg();
  let wasmURL;
  const stop=()=>converter.terminate();
  signal.addEventListener('abort',stop,{once:true});
  try {
    const buffers=[];
    for (const name of ['ffmpeg-core.wasm.part01','ffmpeg-core.wasm.part02']) {
      const response=await fetch(new URL(name,folder),{signal});
      if (!response.ok) throw new Error('No se encontró el conversor de video.');
      buffers.push(await response.arrayBuffer());
    }
    wasmURL=URL.createObjectURL(new Blob(buffers,{type:'application/wasm'}));
    signal.throwIfAborted();
    await converter.load({coreURL:new URL('ffmpeg-core.js',folder).href,wasmURL});
    converter.on('progress',({progress:value})=>progress(Math.round(Math.min(1,Math.max(0,value))*100)));
    await converter.writeFile('input',new Uint8Array(await file.arrayBuffer()));
    signal.throwIfAborted();
    const result=await converter.exec(['-i','input','-an','-vf','scale=min(1280\\,iw):-2','-c:v','libvpx',
      '-pix_fmt','yuv420p','-deadline','realtime','-cpu-used','6','-threads','1','output.webm']);
    if (result!==0) throw new Error('La conversión de video falló.');
    return new Blob([await converter.readFile('output.webm')],{type:'video/webm'});
  } finally {
    stop();
    signal.removeEventListener('abort',stop);
    if (wasmURL) URL.revokeObjectURL(wasmURL);
  }
}
