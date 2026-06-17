# Testing de backend

# Tests unitarios

Para los tests unitarios, el backend `DREAM-ML-backend/GEML` usa la clase nativa de Django, `django.test.TestCase`, que es una subclade de la librería `untittest`. Para calcular la cobertura de los tests se usa la librería `coverage`.

## Instrucciones

* Escribir todos los tests en el archivo `tests.py` de su paquete.
* Separar los tests según la característica o archivo que validen, por ejemplo para el pipeline de clasificación se pueden crear las clases `UtilsTestCase` y `ServicesTestCase` para escribir los tests unitarios.
* Correr los tests del siguiente modo:
    ```bash
    pytest # correr todos los archivos marcados en ./DREAM-ML-backend/GEML/pytest.ini.python_files
    pytest test_mod.py # correr todos los tests en el archivo test_mod.py (busca en pwd)
    pytest testing/ # correr todos los tests en el directorio ./testing (busca en pwd)
    pytest tests/test_mod.py::test_func # correr el test 'test_func'
    ```
    Existen más ejemplos en el link ['How to invoke pytest'](https://pytest.org/en/stable/how-to/usage.html)

Cuando el incremento esté listo para integrarse, correr análisis de cobertura mediante los siguientes comandos:
```bash
coverage run --source='.' -m pytest -v #cobertura para todo el backend

coverage report #imprime en STDOUT
coverage html #escribe reporte en html, bajo la ruta que indica en STDOUT
```