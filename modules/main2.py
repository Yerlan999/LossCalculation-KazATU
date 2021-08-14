import shutil, codecs, re, os, threading
from time import sleep
import os
from tkinter import *
from mytools import MainTrackerClass as main
from tkinter import filedialog, messagebox, HORIZONTAL, END
from tkinter.ttk import Progressbar
from PIL import ImageTk, Image
from pathlib import Path
from tkinter import ttk
from ttkthemes import ThemedTk


def mirror_picture(canvas):

    if main.mirror_state == 1:
        img = Image.open(resource_path(Path(main.all_images_paths[11])))
        resized = img.resize((canvas.winfo_width(), canvas.winfo_height()),
                                 Image.BICUBIC)
        img = ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, anchor=NW, image=img)
        main.current_image = 11
        main.mirror_state = 2
    else:
        img = Image.open(resource_path(Path(main.all_images_paths[1])))
        resized = img.resize((canvas.winfo_width(), canvas.winfo_height()),
                                 Image.BICUBIC)
        img = ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, anchor=NW, image=img)
        main.current_image = 1
        main.mirror_state = 1

    xy_frame.mainloop()


def on_closing():
    if messagebox.askokcancel("Выход из программы", "Вы действительно хотите выйти?"):
        root.destroy()


def help_func():

    help_root = ThemedTk(theme=theme)
    help_root.geometry("600x200")
    help_frame = ttk.Frame(help_root)

    ttk.Label(help_frame, text="Для выполнения расчетов необходимо заполнить все поля приведенные в двух разделах!", width=100, anchor=CENTER, borderwidth=2, relief="groove").pack(side=TOP, fill=BOTH, expand=True)
    ttk.Label(help_frame, text="Все разделы находятся в меню 'Входные данные'", width=100, anchor=CENTER, borderwidth=2, relief="groove").pack(side=BOTTOM, fill=BOTH, expand=True)
    help_frame.pack(side=LEFT, fill=BOTH, expand=True)
    help_root.mainloop()


def validate_numbers(index, numbers):
    # print("Modification at index " + index)
    return globals()["pattern"].match(numbers) is not None

def on_click(event):
    global eb, on_click_id
    eb.delete(0, END)
    eb.unbind('<Button-1>', on_click_id)



def finishing_part(name_podstans):
    inp = name_podstans.strip().replace(" ", "_")
    dir_to_check = f"Результаты_{inp}"
    pa = Path(dir_to_check)
    try:
        if pa.is_dir():
            os.chdir(dir_to_check)
            with codecs.open(f"{dir_to_check.lower()}.txt", "w", "utf-16") as file:
                text_message = u"Здесь будут результаты расчетов(графики) после того как я разберусь с 'Телеграфными уравнениями')))"
                file.write(text_message + u"\n")
        else:
            os.mkdir(dir_to_check)
            os.chdir(dir_to_check)
            with codecs.open(f"{dir_to_check.lower()}.txt", "w", "utf-16") as file:
                text_message = u"Здесь будут результаты расчетов(графики) после того как я разберусь с 'Телеграфными уравнениями')))"
                file.write(text_message + u"\n")
    finally:
        os.chdir(Path("../"))



def subbmit_values(main):

    answer = messagebox.askyesno(title="Расчет", message="Вы уверены что хотите произвести раcчет по указанным данным?")
    if answer == False: return;

    def submit_button():

        # Extracting and preprocessing user input
        try:
            excel_filepath = main.excel_file_path.get()
            excel_filepath_os = Path(excel_filepath)

            name_podstans = main.main_entry_list_vars[0].get().strip()
            name_prisoedin = main.main_entry_list_vars[1].get().strip()
            kolich_prisoed = float(main.main_entry_list_vars[2].get())
            dlina_linii = float(main.main_entry_list_vars[3].get())
            kolich_izmer = float(main.main_entry_list_vars[4].get())
            interval_izmer = float(main.main_entry_list_vars[5].get())
            kolich_garmonik = float(main.main_entry_list_vars[6].get())
            inp_napryazhen_linii = float(main.main_entry_list_vars[7].get())


            if main.edin_imzer_list_vars[0].get() == "м":
                dlina_linii = dlina_linii/1000
            if main.edin_imzer_list_vars[1].get() == 'сек':
                interval_izmer = interval_izmer/60
            if main.edin_imzer_list_vars[2].get() == "В":
                inp_napryazhen_linii = inp_napryazhen_linii/1000

        except:
            messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных воздушной линии")
            main_entry_list_vars = None
            return

        if main.current_image == 1:
            tx, ty, ax, ay, bx, by, cx, cy =  main.xy_grid_list
            list_of_xys = [tx, ty, ax, ay, bx, by, cx, cy]
        if main.current_image == 2:
            tx, ty, a1x, a1y, b1x, b1y, c1x, c1y, a2x, a2y, b2x, b2y, c2x, c2y =  main.xy_grid_list
            list_of_xys = [tx, ty, a1x, a1y, b1x, b1y, c1x, c1y, a2x, a2y, b2x, b2y, c2x, c2y]
        if main.current_image == 3:
            t1x, t1y, t2x, t2y, ax, ay, bx, by, cx, cy =  main.xy_grid_list
            list_of_xys = [t1x, t1y, t2x, t2y, ax, ay, bx, by, cx, cy]
        getted_list_xys = list(map(lambda e: e.get(), list_of_xys))


        try:
            floated_list_xys = list(map(lambda e: float(e), getted_list_xys))
        except:
            messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных для координат фазных проводов и троса")
            return

        # Module for Material Prop. data
        getted_list_matprop = list(map(lambda e: e.get(), main.mat_var_list))

        # Unit conversions happens here
        try:
            floated_list_matprop = list(map(lambda e: float(e), getted_list_matprop))
            if main.edin_imzer_list_vars[3].get() == "кСм/м":
                floated_list_matprop[1] = floated_list_matprop[1]*10**3
                floated_list_matprop[4] = floated_list_matprop[4]*10**3
            if main.edin_imzer_list_vars[3].get() == "См/м":
                floated_list_matprop[1] = floated_list_matprop[1]*10**6
                floated_list_matprop[4] = floated_list_matprop[4]*10**6

            if main.edin_imzer_list_vars[4].get() == "м\u00b2":
                floated_list_matprop[2] = floated_list_matprop[2]*10**6
                floated_list_matprop[5] = floated_list_matprop[5]*10**6
            if main.edin_imzer_list_vars[4].get() == "см\u00b2":
                floated_list_matprop[2] = floated_list_matprop[2]*100
                floated_list_matprop[5] = floated_list_matprop[5]*100
        except:
            messagebox.showerror(title="Ошибка!", message="Недостаточно данных для расчета", detail="Проверьте правильность введенных данных для характеристик материала фазных проводов и троса")
            return

        main.current_frame.destroy()
        main.current_frame = None

        progress_frame = ttk.Frame(root)
        progress_frame.rowconfigure(list(range(0,50)), weight=1)
        progress_frame.columnconfigure(list(range(0,11)), weight=1)

        progress_bar = Progressbar(progress_frame, orient=HORIZONTAL, mode='indeterminate')
        progress_bar.grid(row=20, column=3, columnspan=5, sticky="EWNS")
        progress_bar_label = ttk.Label(progress_frame, text="Обработка данных...", anchor=CENTER, borderwidth=2, relief="groove")
        progress_bar_label.grid(row=19, column=3, columnspan=5, sticky="EWNS")
        progress_bar.start()
        progress_frame.pack(side=BOTTOM, anchor=E, fill=BOTH, expand=True)

        # Clear all entries
        for entry in main.all_entries:
            try:
                entry.delete(0, 'end')
            except:
                print("Something went wrong!")
                pass
        finishing_part(name_podstans)
        final_message = f"Результаты расчетов записаны в папке 'Результаты_{name_podstans}'"


        sleep(5) # EMULATING CALCULATION PROCESS
        progress_bar.stop()
        progress_bar_label.destroy()
        progress_bar.destroy()
        messagebox.showinfo(title="Расчет завершен!", message="Данные успешно обработаны!", detail=final_message)
        root.destroy()

    threading.Thread(target=submit_button).start()




def main_properties(main):

    global eb, on_click_id

    if main.current_frame:
        main.current_frame.pack_forget()
    main.current_frame = main_frame


    ttk.Label(main_frame, text="Общие данные линии", width=55, anchor=CENTER, borderwidth=2, relief="groove").grid(row=0, column=0, columnspan=4, sticky="EWNS")

    for i, (variable, label, var, ed_iz) in enumerate(main.main_entry_list_names, start=1):

        if i == 1:
            ttk.Label(main_frame, text=label, width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")
            e=ttk.Entry(main_frame, width=40, textvariable=eval(variable)); e.grid(row=i, column=1, columnspan=2, sticky="EWNS")
            main.all_entries.append(e)
        elif i == 2:
            ttk.Label(main_frame, text=label, width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")
            eb=ttk.Entry(main_frame, width=40, textvariable=eval(variable)); eb.grid(row=i, column=1, columnspan=2, sticky="EWNS")
            main.all_entries.append(eb)
            eb.insert(0, 'Каждую через ","')
            eb.bind('<Button-1>', on_click)
            on_click_id = eb.bind('<Button-1>', on_click)
        elif ed_iz:
            ttk.Label(main_frame, text=label, width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")
            e=ttk.Entry(main_frame, width=40, validate="key", validatecommand=vcmd, textvariable=eval(variable)); e.grid(row=i, column=1, columnspan=2, sticky="EWNS")
            main.all_entries.append(e)
            OptionMenu(main_frame, eval(var), *ed_iz).grid(row=i, column=3, sticky="EWNS")
        else:
            ttk.Label(main_frame, text=label, width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")
            e=ttk.Entry(main_frame, width=40, validate="key", validatecommand=vcmd, textvariable=eval(variable)); e.grid(row=i, column=1, columnspan=2, sticky="EWNS")
            main.all_entries.append(e)


    ttk.Label(main_frame, text="Характеристика проводов", width=55, anchor=CENTER, borderwidth=2, relief="groove").grid(row=10, column=0, columnspan=4, sticky="EWNS")

    for i, mat_lab in enumerate(main.mat_props, start=12):
        ttk.Label(main_frame, text=mat_lab, width=15, anchor=CENTER, borderwidth=2, relief="groove").grid(row=i, column=0, sticky="EWNS")


    for i, mat_head in enumerate(main.mat_headers, start=1):
        ttk.Label(main_frame, text=mat_head, width=15, anchor=CENTER, borderwidth=2, relief="groove").grid(row=11, column=i, sticky="EWNS")

    for j, provod in enumerate(main.types_cables, start=1):
        for i, character in enumerate(main.mater_charac, start=12):
            me=ttk.Entry(main_frame, width=20, textvariable=eval(provod+character), validate="key", validatecommand=vcmd); me.grid(row=i, column=j, sticky="EWNS")
            main.all_entries.append(me)

    OptionMenu(main_frame, eval(main.mat_edin_izmer[0][0]), *main.mat_edin_izmer[0][1]).grid(row=13, column=3, sticky="EWNS")
    OptionMenu(main_frame, eval(main.mat_edin_izmer[1][0]), *main.mat_edin_izmer[1][1]).grid(row=14, column=3, sticky="EWNS")


    main_frame.pack(side=TOP, anchor=E, fill=BOTH, expand=True)


def resource_path(relative_path):
    try:
        base_path = Path(".")
    except Exception:
        base_path = sys._MEIPASS
    return base_path / relative_path




def draw_picture(canvas):
    global image_set, mirror_button
    canvas.delete("all")

    main.image_set = True
    main.current_image = frame.get()

    if main.current_image == 1:
        main.mirror_button["state"] = "active"
    else:
        main.mirror_button["state"] = "disabled"

    img = Image.open(resource_path(Path(main.all_images_paths[frame.get()])))
    resized = img.resize((canvas.winfo_width(), canvas.winfo_height()),
                             Image.BICUBIC)
    img = ImageTk.PhotoImage(resized)
    canvas.create_image(0, 0, anchor=NW, image=img)


    ttk.Label(xy_frame, text="Y", width=1, anchor=CENTER, borderwidth=2, relief="groove").grid(row=1, column=3, sticky="EWNS")
    ttk.Label(xy_frame, text="X", width=1, anchor=CENTER, borderwidth=2, relief="groove").grid(row=1, column=2, sticky="EWNS")


    # ROPE LABELS
    for cur_lab in main.xy_labels:
        if cur_lab:
            cur_lab.destroy()
    main.xy_labels= []


    for i, label in enumerate(main.xyframe_label_dict[frame.get()], start=2):
        l=ttk.Label(xy_frame, text=label, width=4, anchor=CENTER, borderwidth=2, relief="groove"); l.grid(row=i, column=1, sticky="EWNS")
        main.xy_labels.append(l)


    # XY ENTRIES
    for xy_curr_entries in main.xy_grid_list:
        if xy_curr_entries:
            xy_curr_entries.destroy()
    main.xy_grid_list = []

    for i, rope in enumerate(main.xyframe_label_dict[frame.get()], start=2):
        for j, xy in enumerate(main.xy_headers, start=2):
            inp_var = (rope+xy).lower()
            exec(inp_var + "=StringVar()")
            main.xy_var_list.append(eval(inp_var))
            xye=ttk.Entry(xy_frame, textvariable=eval(inp_var), width=1, validate="key", validatecommand=vcmd); xye.grid(row=i, column=j, sticky="EWNS")
            main.xy_grid_list.append(xye)



    xy_frame.mainloop()


def type_razche(xy_frame):
        if inp_faza_razshep.get():
            r1=ttk.Radiobutton(xy_frame, text="(2) Двойная", value=1, variable=inp_type_razchep); r1.grid(row=11, column=1, columnspan=3, sticky="WNS")
            r2=ttk.Radiobutton(xy_frame, text="(3) Треугольник", value=2, variable=inp_type_razchep); r2.grid(row=12, column=1, columnspan=3, sticky="WNS")
            r3=ttk.Radiobutton(xy_frame, text="(4) Квардрат", value=3, variable=inp_type_razchep); r3.grid(row=13, column=1, columnspan=3, sticky="WNS")
            main.rad_but_list = [r1, r2, r3]
        else:
            for rad_but in main.rad_but_list:
                rad_but.destroy()



def xy_properties(main):
    global canvas
    if main.current_frame:
        main.current_frame.pack_forget()
    main.current_frame = xy_frame


    ttk.Label(xy_frame, text="Характеристика опор", width=75, anchor=CENTER, borderwidth=2, relief="groove").grid(row=0, column=0, columnspan=4, sticky="EWNS")
    canvas = Canvas(xy_frame, width = 281, height = 313, bg='white')
    canvas.grid(row=1, rowspan=9, column=0, sticky="EWNS")

    main.mirror_button = ttk.Button(xy_frame, text="<>", command=lambda:mirror_picture(canvas), cursor='hand2')
    main.mirror_button.grid(row=9, column=0, sticky="SE")
    main.mirror_button["state"] = "disabled"

    ttk.Label(xy_frame, text="Тип опоры", width=25, anchor=CENTER, borderwidth=2, relief="groove").grid(row=10, column=0, columnspan=1, sticky="EWNS")

    ttk.Checkbutton(xy_frame, text="Фаза разщеплена",
                 onvalue = True, offvalue = False,
                 variable=inp_faza_razshep,
                 command = lambda: type_razche(xy_frame), cursor='hand2'
                ).grid(row=10, column=1, columnspan=3, sticky="WNSE")


    def resize_bg(event):
        global bgg, resized, bg2, canvas
        if not main.image_set:
            return
        canvas.delete("all")

        bgg = Image.open(main.all_images_paths[main.current_image])
        resized = bgg.resize((canvas.winfo_width(), canvas.winfo_height()),
                             Image.BICUBIC)

        bg2 = ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, image=bg2, anchor='nw')

    ttk.Radiobutton(xy_frame, text="Одноцепная промежуточная", variable=frame, value=1, command=lambda:draw_picture(canvas), cursor='hand2').grid(row=11, column=0, sticky="WNS")
    ttk.Radiobutton(xy_frame, text="Двухцепная промежуточная", variable=frame, value=2, command=lambda:draw_picture(canvas), cursor='hand2').grid(row=12, column=0, sticky="WNS")
    ttk.Radiobutton(xy_frame, text="Портальная промежуточная", variable=frame, value=3, command=lambda:draw_picture(canvas), cursor='hand2').grid(row=13, column=0, sticky="WNS")


    def fileinput():
        filename = filedialog.askopenfilename(filetypes=[("Файлы EXCEL", ".xlsx .xls"), ("Все файлы","*.*")])
        excel_filepath.insert(END, filename) # add this


    label_excel = ttk.Label(xy_frame, text="Путь к EXCEL файлу", width=25, anchor=CENTER, borderwidth=2, relief="groove")
    label_excel.grid(row=14, column=0, sticky="EWNS")

    main.excel_file_path = StringVar()
    excel_filepath = ttk.Entry(xy_frame, width=5, textvariable=main.excel_file_path)
    main.all_entries.append(main.excel_file_path)
    excel_filepath.grid(row=14, column=1, sticky="EWNS")

    select_button_excel = ttk.Button(xy_frame, text="Выбрать", command=fileinput, cursor='hand2')
    select_button_excel.grid(row=14, column=2, columnspan=2, sticky="EWNS")

    xy_frame.bind("<Configure>", resize_bg)
    xy_frame.pack(side=TOP, anchor=E, fill=BOTH, expand=True)
    xy_frame.mainloop()

main.image_set = False

theme = 'breeze'

root = ThemedTk(theme=theme)
root.geometry("600x600+0+0")


pattern = re.compile(r'^([\.\d]*)$')
# globals()["pattern"] = pattern
vcmd = (root.register(validate_numbers), "%i", "%P")



frame = IntVar()
inp_faza_razshep = BooleanVar()
inp_type_razchep = IntVar()


main_frame = ttk.Frame(root)
main_frame.rowconfigure(list(range(0,17)), weight=2)
main_frame.columnconfigure(list(range(0,4)), weight=1)
main_frame.columnconfigure(3, weight=0)


xy_frame = ttk.Frame(root)
xy_frame.rowconfigure(list(range(0,19)), weight=2)
xy_frame.columnconfigure(list(range(1,4)), weight=1)
xy_frame.columnconfigure(0, weight=2)


main.list_of_frames = [xy_frame, main_frame]


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Общие характеристики линии", command=lambda:main_properties(main))
filemenu.add_command(label="Характеристика опоры", command=lambda:xy_properties(main))
filemenu.add_separator()
filemenu.add_command(label="Рассчитать потери", command=lambda:subbmit_values(main))
menubar.add_cascade(label="Входные данные", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Инструкция программы", command=help_func)
helpmenu.add_command(label="Выйти из программы", command=on_closing)
menubar.add_cascade(label="О программе", menu=helpmenu)

root.config(menu=menubar)

root.protocol("WM_DELETE_WINDOW", on_closing)

for main_var, *rest in main.main_entry_list_names:
    exec(main_var + "=StringVar()")
    main.main_entry_list_vars.append(eval(main_var))


for main_var, def_val in main.edin_imzer_list_names:
    exec(main_var + "=StringVar()")
    eval(main_var).set(def_val)
    main.edin_imzer_list_vars.append(eval(main_var))


for provod in main.types_cables:
    for character in main.mater_charac:
        exec(provod+character + "=StringVar()")
        main.mat_var_list.append(eval(provod+character))

root.mainloop()

os.system("taskkill /F /IM python3.8.exe /T")
