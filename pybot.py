from aiogram import Bot, types,executor
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types.message import ContentType
from mainCopy import process_image
from aiogram.dispatcher.filters.state import State, StatesGroup

cur_id = 0
token = '5223909518:AAF6FKxnFn8cDUXMIunBIxguPzFYvnyhH7c'

storage = MemoryStorage()
bot = Bot(token=token)
dp = Dispatcher(bot,storage=storage)



start_message = 'Привіт! 😊 Це бот від команди Примат І ПСИ. Я допоможу тобі зі складанням будь якого пазлу☺️👍🏽. Щоб розпочати просто надійшли мені фото оригінал фото та трохи почекай, я надішлю тобі інструкції по його складанню! 🤪🤪'




class States(StatesGroup):
    START_STATE = State()
    PIC_1_GOT = State()


@dp.message_handler(commands=['start'], state='*')
async def process_start_command(message: types.Message):
        await States.START_STATE.set()
        await message.reply(start_message)


@dp.message_handler(content_types=ContentType.DOCUMENT,state=States.START_STATE)
async def get_photos1(message: types.Message):
        await States.PIC_1_GOT.set()
        print(message.message_id)
        await message.reply('Тепер пазл!')
        full_image_fn = f'{message.from_user.id}main.jpg'
        print(message.document)
        await message.document.download(full_image_fn)


@dp.message_handler(content_types=ContentType.DOCUMENT,state=States.PIC_1_GOT)
async def get_photos2(message: types.Message):
        await message.reply('Хвилиночку!')
        print(message.message_id)
        puzzle_fn = f'{message.from_user.id}puzzle.jpg'
        full_image_fn = f'{message.from_user.id}main.jpg'
        await message.document.download(puzzle_fn)
        res = process_image(full_image_fn,puzzle_fn)
        await States.START_STATE.set()
        for ph_fn in res:
                await bot.send_photo(message.chat.id,photo=open(ph_fn, 'rb'))


@dp.message_handler()
async def change_state(message: types.Message):
        await message.reply('Надішли фото, будь ласка, я більше нічого не вмію :)')



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)