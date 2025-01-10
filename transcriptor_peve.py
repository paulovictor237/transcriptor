import os
import glob
import whisper
import warnings
import time

# Suprimir avisos específicos do Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def formatar_tempo(segundos):
    """Converte segundos em formato legível."""
    return time.strftime('%H:%M:%S', time.gmtime(segundos))

def transcrever_audio(audio_path, model):
    """Transcreve um arquivo de áudio usando o Whisper."""
    try:
        print("Iniciando transcrição...")
        tempo_inicio = time.time()
        
        # Realizar transcrição
        result = model.transcribe(audio_path, language='pt', fp16=False)
        
        tempo_total = time.time() - tempo_inicio
        print(f"Tempo de transcrição: {formatar_tempo(tempo_total)}")
        
        return result["text"]
        
    except Exception as e:
        return f"Erro durante a transcrição: {str(e)}"

def quebrar_linhas(texto, largura=80):
    """Quebra o texto em linhas de no máximo 'largura' caracteres."""
    linhas = []
    for paragrafo in texto.split('\n'):
        while len(paragrafo) > largura:
            corte = paragrafo[:largura].rfind(' ')
            if corte == -1:
                corte = largura
            linhas.append(paragrafo[:corte])
            paragrafo = paragrafo[corte:].lstrip()
        linhas.append(paragrafo)
    return '\n'.join(linhas)

def main():
    # Configuração do modelo Whisper
    print("Carregando modelo Whisper (isso pode levar alguns minutos na primeira vez)...")
    model = whisper.load_model("base", device="cpu")
    print("Modelo carregado com sucesso!")
    
    # Encontrar todos os arquivos MP4 e MP3 no diretório atual
    arquivos_audio = glob.glob("*.mp4") + glob.glob("*.mp3")
    
    if not arquivos_audio:
        print("Nenhum arquivo MP4 ou MP3 encontrado no diretório atual.")
        return
        
    print(f"Encontrados {len(arquivos_audio)} arquivos de áudio.")
    
    for i, audio_path in enumerate(arquivos_audio, 1):
        print(f"\nProcessando arquivo {i} de {len(arquivos_audio)}: {audio_path}")
        
        # Criar nome para arquivo de saída
        nome_base = os.path.splitext(audio_path)[0]
        arquivo_saida = f"{nome_base}.txt"
        
        try:
            # Transcrever áudio
            transcricao = transcrever_audio(audio_path, model)
            transcricao_formatada = quebrar_linhas(transcricao)
            
            # Salvar transcrição
            with open(arquivo_saida, 'w', encoding='utf-8') as f:
                f.write(transcricao_formatada)
            
            print(f"✓ Transcrição salva com sucesso em: {arquivo_saida}")
            
        except Exception as e:
            print(f"✗ Erro ao processar {audio_path}: {str(e)}")
        
        print("-------------------")

if __name__ == "__main__":
    main()