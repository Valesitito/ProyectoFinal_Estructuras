# =============================================================================
# PROYECTO FINAL: ASISTENTE VIRTUAL POR GESTOS
# MATERIA: ESTRUCTURA DE DATOS Y ALGORITMOS
# SEMESTRE: 3
# 
# Descripción:
# Sistema de control por visión artificial que implementa las estructuras
# vistas en clase: Listas Dobles, Árboles BST, Pilas, Colas y Grafos.
#
# Referencias:
# - Detección de manos: Documentación oficial MediaPipe / OpenCV
# - Automatización de teclado: Documentación oficial PyAutoGUI
# - Serialización: Documentación oficial Pickle
# 
# Integrantes:
# - Jorge Carlos Vales Martínez (00545512)
# - Daniela Alejandra Goméz Méndez (00557226)
# =============================================================================

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import pickle
import os

# -----------------------------------------------------------------------------
# TEMA 4: ÁRBOLES (BST - Binary Search Tree)
# Implementamos un Árbol Binario para indexar los gestos.
# Justificación: Buscar en una lista es O(n), buscar en un árbol balanceado es O(log n).
# -----------------------------------------------------------------------------
class NodoArbol:
    def __init__(self, gesto):
        self.data = gesto
        self.izq = None  # Apuntador hijo izquierdo
        self.der = None  # Apuntador hijo derecho

class ArbolGestos:
    def __init__(self):
        self.raiz = None

    def insertar(self, gesto):
        if not self.raiz:
            self.raiz = NodoArbol(gesto)
        else:
            self._insertar_recursivo(self.raiz, gesto)

    # Usamos recursividad porque es más limpio para la lógica de árboles
    def _insertar_recursivo(self, nodo, nuevo_gesto):
        # Ordenamos alfabéticamente por nombre
        if nuevo_gesto.nombre < nodo.data.nombre:
            if nodo.izq is None: nodo.izq = NodoArbol(nuevo_gesto)
            else: self._insertar_recursivo(nodo.izq, nuevo_gesto)
        else:
            if nodo.der is None: nodo.der = NodoArbol(nuevo_gesto)
            else: self._insertar_recursivo(nodo.der, nuevo_gesto)

    def buscar(self, nombre):
        """Busca un gesto por nombre de forma eficiente."""
        return self._buscar_logic(self.raiz, nombre)

    def _buscar_logic(self, nodo, nombre):
        if not nodo: return None
        if nodo.data.nombre == nombre: return nodo.data
        
        # Decidimos si ir a izquierda o derecha
        if nombre < nodo.data.nombre: return self._buscar_logic(nodo.izq, nombre)
        return self._buscar_logic(nodo.der, nombre)

    def vaciar(self):
        # Reiniciamos el árbol
        self.raiz = None

# -----------------------------------------------------------------------------
# TEMA 2: LISTAS LIGADAS (Doblemente Ligadas)
# Usamos lista doble para poder tener referencia al anterior y siguiente.
# También mantenemos un puntero 'cola' para inserción O(1) al final.
# -----------------------------------------------------------------------------
class NodoLista:
    def __init__(self, dato):
        self.dato = dato
        self.siguiente = None
        self.anterior = None # Puntero extra para cumplir requisito de Lista Doble

class ListaDoble:
    def __init__(self):
        self.cabeza = None
        self.cola = None

    def agregar(self, nuevo_gesto):
        nuevo_nodo = NodoLista(nuevo_gesto)
        if not self.cabeza:
            self.cabeza = nuevo_nodo
            self.cola = nuevo_nodo
        else:
            # Aprovechamos el puntero 'cola' para no recorrer toda la lista
            ultimo = self.cola
            ultimo.siguiente = nuevo_nodo
            nuevo_nodo.anterior = ultimo
            self.cola = nuevo_nodo

    def eliminar_por_nombre(self, nombre):
        # Recorrido lineal O(n)
        actual = self.cabeza
        while actual:
            if actual.dato.nombre == nombre:
                # Re-enlazamos punteros (bypass del nodo)
                if actual.anterior: actual.anterior.siguiente = actual.siguiente
                else: self.cabeza = actual.siguiente
                
                if actual.siguiente: actual.siguiente.anterior = actual.anterior
                else: self.cola = actual.anterior
                return True
            actual = actual.siguiente
        return False

    def esta_vacia(self):
        return self.cabeza is None

    # Método mágico para poder hacer "for x in lista"
    def __iter__(self):
        actual = self.cabeza
        while actual:
            yield actual.dato
            actual = actual.siguiente

# -----------------------------------------------------------------------------
# TEMA 5: GRAFOS (Máquina de Estados / Contextos)
# Implementación de un Grafo Dirigido usando Listas de Adyacencia (Diccionarios).
# - Vértices (Nodos) = Modos (Normal, Zoom, etc.)
# - Aristas = Transiciones o Acciones disponibles en ese modo
# -----------------------------------------------------------------------------
class GrafoContextual:
    def __init__(self):
        # Definición de Nodos (Vértices)
        self.modos = {
            'normal': {'color': (0, 255, 0), 'desc': 'Navegacion Normal'},
            'zoom': {'color': (255, 255, 0), 'desc': 'Modo Zoom'},
            'anotacion': {'color': (255, 0, 255), 'desc': 'Modo Dibujo'},
            'video': {'color': (0, 255, 255), 'desc': 'Control Video'}
        }
        self.modo_actual = 'normal'
        
        # Definición de Aristas (Conexiones/Acciones)
        self.acciones = { 'normal': {}, 'zoom': {}, 'anotacion': {}, 'video': {} }
        self._init_defaults()
    
    def _init_defaults(self):
        # Configuramos las aristas por defecto
        self.acciones['normal'] = {'derecha': 'right', 'izquierda': 'left', 'arriba': 'home', 'abajo': 'end'}
        self.acciones['zoom'] = {'derecha': 'zoom_pan_right', 'mas': 'zoom_in', 'menos': 'zoom_out'}
        self.acciones['video'] = {'puño': 'space', 'derecha': 'video_forward'}
    
    def cambiar_modo(self, nuevo):
        # Transición en el grafo
        if nuevo in self.modos: self.modo_actual = nuevo

    def ciclar_modo(self):
        # Recorrido circular de los nodos del grafo
        lista = list(self.modos.keys())
        idx = lista.index(self.modo_actual)
        self.cambiar_modo(lista[(idx + 1) % len(lista)])

    def get_accion(self, gesto):
        # Retorna la arista correspondiente al gesto en el nodo actual
        return self.acciones[self.modo_actual].get(gesto)

    def get_info(self):
        return self.modos[self.modo_actual]

# -----------------------------------------------------------------------------
# TEMA 3: PILAS, COLAS Y ESTRUCTURAS AUXILIARES
# -----------------------------------------------------------------------------
class Pila: 
    # Implementación LIFO para el historial de "Undo"
    def __init__(self): self.items = []
    def push(self, x): self.items.append(x)
    def pop(self): return self.items.pop() if self.items else None

class Cola: 
    # Implementación FIFO para buffer de comandos
    def __init__(self): self.items = []
    def encolar(self, x): self.items.insert(0, x)
    def desencolar(self): return self.items.pop() if self.items else None

class GestoPersonalizado:
    # Esta clase actúa como un "Struct" avanzado
    def __init__(self, nombre, accion):
        self.nombre = nombre
        self.accion = accion
        self.patron_puntos = []

    def grabar_patron(self, puntos):
        # Normalización matemática para que no importe si la mano está cerca o lejos
        arr = np.array(puntos)
        base = arr[0] # Usamos la muñeca como origen (0,0)
        arr -= base
        max_d = np.max(np.linalg.norm(arr, axis=1))
        if max_d > 0: arr /= max_d
        self.patron_puntos = arr.tolist()

class PerfilUsuario:
    def __init__(self, nombre):
        self.nombre = nombre
        # ESTRUCTURA HÍBRIDA:
        # Usamos Lista para orden secuencial y Árbol para búsquedas rápidas.
        self.gestos = ListaDoble()
        self.indice = ArbolGestos()

    def agregar_gesto(self, g):
        self.gestos.eliminar_por_nombre(g.nombre) # Evitar duplicados
        self.gestos.agregar(g)
        
        # Reconstruimos el árbol (método simple para mantener consistencia)
        self.indice.vaciar()
        for gesto in self.gestos:
            self.indice.insertar(gesto)

# PERSISTENCIA DE DATOS
DB_FILE = "database_perfiles_v3.pkl"

def guardar_todo(perfiles, grafo):
    data = {'perfiles': perfiles, 'grafo': grafo.acciones}
    try:
        with open(DB_FILE, 'wb') as f: pickle.dump(data, f)
        print(">> Base de datos (Pickle) actualizada.")
    except Exception as e:
        print(f"Error al guardar: {e}")

def cargar_todo():
    if not os.path.exists(DB_FILE): return {}, None
    try:
        with open(DB_FILE, 'rb') as f:
            data = pickle.load(f)
            # Soporte legacy para versiones anteriores de la DB
            perfiles = data.get('perfiles', {}) if isinstance(data, dict) else data
            grafo_data = data.get('grafo', None) if isinstance(data, dict) else None
            
            # RECONSTRUCCIÓN DE ESTRUCTURAS
            # Fix necesario porque pickle a veces pierde las referencias de clases nuevas
            for p in perfiles.values():
                lista_vieja = []
                if isinstance(p.gestos, list): lista_vieja = p.gestos
                else: 
                    for g in p.gestos: lista_vieja.append(g)
                
                # Reinicializamos las estructuras de datos
                p.gestos = ListaDoble()
                p.indice = ArbolGestos()
                for g in lista_vieja: p.agregar_gesto(g)
            
            return perfiles, grafo_data
    except: return {}, None

# -----------------------------------------------------------------------------
# MENÚS (INTERFAZ DE CONSOLA)
# -----------------------------------------------------------------------------
def gestionar_gestos(perfil, todos_perfiles, grafo):
    while True:
        print("\n" + "="*60)
        print(f"GESTIONAR GESTOS DEL PERFIL: {perfil.nombre}")
        print("="*60)
        
        if perfil.gestos.esta_vacia():
            print("  (No hay gestos grabados)")
        else:
            print("Gestos actuales (Recorriendo Lista Doble):")
            for g in perfil.gestos:
                print(f"  - {g.nombre}  -->  {g.accion}")

        print("\nOPCIONES:")
        print("  (g) Grabar nuevo gesto")
        print("  (b) Buscar gesto (Demo Arbol BST)")
        print("  (e) Eliminar gesto")
        print("  (s) Salir")
        
        op = input("\nSelecciona: ").lower()

        if op == 'g':
            nombre = input("Nombre para el gesto: ")
            accion = input(f"Accion (tecla) para '{nombre}': ")
            print("\n>> INSTRUCCION: Regresa a la camara, haz el gesto y presiona 's'.")
            return GestoPersonalizado(nombre, accion)
            
        elif op == 'b':
            busq = input("Nombre a buscar: ")
            # Aquí demostramos el uso del Árbol de Búsqueda
            res = perfil.indice.buscar(busq)
            if res: print(f"¡Encontrado en Árbol!: {res.nombre} ejecuta {res.accion}")
            else: print("No existe en el índice.")
            
        elif op == 'e':
            nom = input("Nombre a borrar: ")
            perfil.gestos.eliminar_por_nombre(nom)
            # Actualizamos el árbol para que coincida con la lista
            perfil.indice.vaciar()
            for g in perfil.gestos: perfil.indice.insertar(g)
            guardar_todo(todos_perfiles, grafo)
            print("Eliminado.")

        elif op == 's': break
    return None

def gestionar_modos(grafo, todos_perfiles):
    while True:
        print("\n" + "="*60)
        print("CONFIGURACION DE MODOS (GRAFO)")
        print("="*60)
        print(f"Modo Actual (Nodo): {grafo.modo_actual.upper()}")
        print("\nOPCIONES:")
        print("  (1) Normal  (2) Zoom  (3) Anotacion  (4) Video")
        print("  (v) Ver acciones (Aristas) del modo actual")
        print("  (s) Salir")
        
        op = input("\nOpcion: ").lower()
        
        if op in ['1','2','3','4']:
            modos = ['normal', 'zoom', 'anotacion', 'video']
            grafo.cambiar_modo(modos[int(op)-1])
            print(f"Modo cambiado a: {grafo.modo_actual}")
        elif op == 'v':
            print(f"\nAcciones en {grafo.modo_actual}:")
            print(grafo.acciones[grafo.modo_actual])
        elif op == 's': break

def menu_principal_perfiles(perfiles, grafo):
    while True:
        print("\n" + "="*60)
        print("SISTEMA DE GESTOS - SELECCION DE PERFIL")
        print("="*60)
        
        lista = list(perfiles.keys())
        if not lista: print("  (Sin perfiles)")
        else:
            for p in lista: print(f"  - {p}")
            
        print("\n(n) Nuevo Perfil  |  (c) Cargar Perfil")
        op = input("Selecciona: ").lower()
        
        if op == 'n':
            nom = input("Nombre nuevo: ")
            perfiles[nom] = PerfilUsuario(nom)
            return perfiles[nom]
        elif op == 'c':
            nom = input("Nombre a cargar: ")
            if nom in perfiles: return perfiles[nom]
            print("No existe.")
        guardar_todo(perfiles, grafo)

# -----------------------------------------------------------------------------
# LÓGICA PRINCIPAL Y VISIÓN COMPUTACIONAL
# -----------------------------------------------------------------------------

def encontrar_gesto_estricto(puntos, perfil):
    """
    ALGORITMO MEJORADO (V2.1):
    Antes usábamos solo el promedio de error (Euclidiana), pero eso fallaba
    si solo un dedo estaba mal (ej. Palma vs 4 dedos).
    Ahora validamos también el "Error Máximo" de un solo punto.
    """
    if not puntos or perfil.gestos.esta_vacia(): return None, 999
    
    # 1. Normalización (Invariante a escala)
    arr = np.array(puntos)
    base = arr[0]
    arr -= base
    mx = np.max(np.linalg.norm(arr, axis=1))
    if mx > 0: arr /= mx
    
    mejor_gesto = None
    menor_score = 999
    
    # UMBRALES DE TOLERANCIA
    UMBRAL_PROMEDIO = 0.15  # Tolerancia general
    UMBRAL_MAX_PUNTO = 0.25 # Si UN punto está así de lejos, rechazamos.
    
    for g in perfil.gestos:
        if not g.patron_puntos: continue
        patron = np.array(g.patron_puntos)
        
        # Calculamos distancia de cada uno de los 21 puntos
        distancias = np.linalg.norm(arr - patron, axis=1)
        
        error_promedio = np.mean(distancias)
        error_maximo = np.max(distancias) # El dedo que más se equivoca
        
        # Validación Estricta:
        if error_promedio < UMBRAL_PROMEDIO and error_maximo < UMBRAL_MAX_PUNTO:
            
            # Score ponderado: Castigamos más si hay un dedo muy chueco
            score = (error_promedio * 0.7) + (error_maximo * 0.3)
            
            if score < menor_score:
                menor_score = score
                mejor_gesto = g
                
    return mejor_gesto, menor_score

def main():
    print("Iniciando camara y sistemas...")
    
    # Carga de Base de Datos
    todos_perfiles, grafo_data = cargar_todo()
    grafo = GrafoContextual()
    if grafo_data: grafo.acciones = grafo_data
    
    # Menú inicial
    perfil_actual = menu_principal_perfiles(todos_perfiles, grafo)
    
    # Configuración de MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    
    # Estructuras de ejecución
    pila_undo = Pila()
    cola_cmd = Cola()
    modo_grabacion = False
    gesto_a_grabar = None
    
    t_ultimo = 0
    COOLDOWN = 1.2 # Segundos de espera entre acciones
    
    print("\n" + "="*70)
    print(f"SISTEMA LISTO. PERFIL: {perfil_actual.nombre}")
    print("CONTROLES: 'm'=Menu, 'x'=Modos, 'TAB'=Ciclar Modo, 'q'=Salir")
    print("="*70)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        
        # Espejo y conversión de color
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        pts_mano = []
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            for lm in hand.landmark: pts_mano.append([lm.x, lm.y])

        # --- INTERFAZ GRÁFICA (Estilo Clásico V2) ---
        # 1. Indicador de Perfil
        cv2.putText(frame, f"Perfil: {perfil_actual.nombre}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
        # 2. Indicador de Modo (Color según el nodo del grafo)
        info = grafo.get_info()
        cv2.putText(frame, f"Modo: {info['desc']}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, info['color'], 2)

        if modo_grabacion:
            cv2.putText(frame, f"GRABANDO: {gesto_a_grabar.nombre}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Presiona 's' para salvar", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Usamos la nueva función con validación de "Peor Dedo"
            gesto, score = encontrar_gesto_estricto(pts_mano, perfil_actual)
            
            reconocido = gesto is not None
            
            color_st = (0, 255, 0) if reconocido else (0, 255, 255)
            status_txt = "RECONOCIDO" if reconocido else "ESPERANDO..."
            
            cv2.putText(frame, f"Estado: {status_txt}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_st, 2)
            
            if reconocido:
                cv2.putText(frame, f"Gesto: {gesto.nombre} (Err: {score:.3f})", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_st, 2)
                
                # Ejecución de Acciones
                if time.time() - t_ultimo > COOLDOWN:
                    # Buscamos la arista en el grafo
                    accion = grafo.get_accion(gesto.nombre)
                    if not accion: accion = gesto.accion # Fallback
                    
                    if accion == '_undo_':
                        prev = pila_undo.pop()
                        if prev: 
                            print(f"Undo: {prev}")
                            if prev == 'right': pyautogui.press('left')
                            elif prev == 'left': pyautogui.press('right')
                    else:
                        print(f">> Ejecutando: {accion}")
                        
                        # Manejo de acciones especiales vs teclas normales
                        if 'zoom' in accion or 'video' in accion:
                            if 'right' in accion: pyautogui.press('right')
                            elif 'left' in accion: pyautogui.press('left')
                            elif 'space' in accion: pyautogui.press('space')
                        else:
                            pyautogui.press(accion)
                            
                        pila_undo.push(accion)
                    
                    t_ultimo = time.time()

        cv2.imshow('Asistente Gestos - Proyecto Final', frame)
        
        # --- INPUTS ---
        k = cv2.waitKey(5) & 0xFF
        if k == ord('q'): break
        
        if k == ord('m') and not modo_grabacion:
            nuevo = gestionar_gestos(perfil_actual, todos_perfiles, grafo)
            if nuevo:
                gesto_a_grabar = nuevo
                modo_grabacion = True
                
        if k == ord('s') and modo_grabacion:
            if pts_mano:
                gesto_a_grabar.grabar_patron(pts_mano)
                perfil_actual.agregar_gesto(gesto_a_grabar)
                guardar_todo(todos_perfiles, grafo)
                print(f"Gesto guardado (Validación estricta activada).")
                modo_grabacion = False
            else:
                print("¡Error! No veo tu mano.")

        if k == ord('x'): gestionar_modos(grafo, todos_perfiles)
        if k == 9: grafo.ciclar_modo() # TAB Key

    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente.")

if __name__ == "__main__":
    main()